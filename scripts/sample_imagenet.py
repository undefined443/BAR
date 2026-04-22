import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
from types import SimpleNamespace
import torch
import torch.distributed as dist
import os
import time
import wandb
import json
from tqdm import tqdm
from utils.logger import setup_logger
from utils.train_utils import create_dataloader, get_pretrained_tokenizer
from utils.eval_utils import load_refs_from_wds, compute_metrics
from modeling.generator import BAR


def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def init_wandb_run(config):
    metadata_path = (
        Path(config.experiment.generator_checkpoint).parent.parent / "metadata.json"
    )
    wandb_run_id = json.loads(metadata_path.read_text()).get("wandb_run_id")
    return wandb.init(
        project=config.experiment.project,
        name=config.experiment.name,
        id=wandb_run_id,
        resume="allow",
        config=OmegaConf.to_container(config, resolve=True),
    )


def log_metrics_to_wandb(run, metrics_by_order):
    combined = {
        f"eval/{k}/{order}": v
        for order, metrics in metrics_by_order.items()
        for k, v in metrics.items()
    }
    run.log(combined)


def main():
    config = get_config_cli()
    per_proc_batch_size = config.experiment.get("per_proc_batch_size", 125)
    sample_folder_dir = config.experiment.output_dir
    seed = config.experiment.get("random_seed", 42)
    sample_speed_benchmark = config.experiment.get("sample_speed_benchmark", False)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    # setup DDP.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = local_rank
    seed = seed + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    tokenizer = get_pretrained_tokenizer(config)
    tokenizer.to(device)
    tokenizer_encode_fn = tokenizer.encode

    generator = BAR(config)
    checkpoint = torch.load(config.experiment.generator_checkpoint, map_location="cpu")
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    generator.load_state_dict(cleaned_state_dict)
    generator.eval()
    generator.requires_grad_(False)
    generator.to(device)

    # Log generator model size
    if rank == 0:
        total_params = sum(p.numel() for p in generator.parameters())
        print(f"Generator parameters: {total_params:,}")

    if not sample_speed_benchmark:
        if rank == 0:
            os.makedirs(sample_folder_dir, exist_ok=True)
            print(f"Saving .txt samples at {sample_folder_dir}")
    else:
        if rank == 0:
            print("Speed benchmark mode: skipping image saving and npz creation")
            print("GPU warmup: first 10 batches will be excluded from timing")
    dist.barrier()

    n = per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    accelerator = SimpleNamespace(
        num_processes=world_size,
        process_index=rank,
        device=torch.device(f"cuda:{local_rank}"),
    )

    logger = setup_logger(
        name="SAMPLE",
        log_level="INFO",
        output_file=f"{sample_folder_dir}/log{accelerator.process_index}.txt",
        use_accelerate=False,
    )

    _, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Benchmark variables
    warmup_batches = 10 if sample_speed_benchmark else 0
    start_time = None
    benchmark_captions = 0
    preds_raster = {}
    preds_random = {}

    total_batches = 5000 // global_batch_size
    progress = tqdm(
        eval_dataloader,
        total=total_batches,
        desc="Sampling",
        unit="batch",
        disable=rank != 0,
    )
    for batch_idx, batch in enumerate(progress):
        # Start timing after warmup batches
        if sample_speed_benchmark and batch_idx == warmup_batches:
            torch.cuda.synchronize()
            start_time = time.time()

        tokens_allocation = config.model.generator.mbm_head.get(
            "tokens_allocation", None
        )

        captions = batch["caption"]
        images = batch["image"].to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )
        image_ids = [int(k) for k in batch["__key__"]]

        with torch.no_grad():
            input_tokens, conditions = tokenizer_encode_fn(captions, images)
            input_tokens = input_tokens.reshape(len(captions), -1)

        generate_kwargs = dict(
            condition=conditions,
            guidance_scale=config.model.generator.guidance_scale,
            randomize_temperature=config.model.generator.mbm_head.randomize_temperature,
            kv_cache=True,
            tokens_allocation=tokens_allocation,
        )

        raster_captions = tokenizer.decode_tokens(
            generator.generate(**generate_kwargs, sample_with_random_order=False)
        )
        random_captions = tokenizer.decode_tokens(
            generator.generate(**generate_kwargs, sample_with_random_order=True)
        )

        preds_raster.update(
            {image_id: [cap] for image_id, cap in zip(image_ids, raster_captions)}
        )
        preds_random.update(
            {image_id: [cap] for image_id, cap in zip(image_ids, random_captions)}
        )

        if not sample_speed_benchmark:
            raster_dir = os.path.join(sample_folder_dir, "raster")
            random_dir = os.path.join(sample_folder_dir, "random")
            os.makedirs(raster_dir, exist_ok=True)
            os.makedirs(random_dir, exist_ok=True)
            for image_id, raster_cap, random_cap in zip(
                image_ids, raster_captions, random_captions
            ):
                filename = f"{image_id:06d}.txt"
                with open(f"{raster_dir}/{filename}", "w") as f:
                    f.write(raster_cap)
                with open(f"{random_dir}/{filename}", "w") as f:
                    f.write(random_cap)

        # Count captions after warmup for benchmark
        if sample_speed_benchmark and batch_idx >= warmup_batches:
            benchmark_captions += global_batch_size

    all_raster_list = [None] * world_size
    all_random_list = [None] * world_size
    dist.all_gather_object(all_raster_list, preds_raster)
    dist.all_gather_object(all_random_list, preds_random)
    if rank == 0:
        merged_raster = {k: v for d in all_raster_list for k, v in d.items()}
        merged_random = {k: v for d in all_random_list for k, v in d.items()}
        refs = load_refs_from_wds(config.dataset.params.eval_shards_path_or_url)

        metrics_by_order = {}
        for order, merged_preds in [
            ("Raster Order", merged_raster),
            ("Random Order", merged_random),
        ]:
            metrics = compute_metrics(merged_preds, refs)
            logger.info(
                f"Metrics ({order}): "
                + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            )
            metrics_by_order[order] = metrics

        if config.training.enable_wandb:
            run = init_wandb_run(config)
            log_metrics_to_wandb(run, metrics_by_order)
            run.finish()

    # Make sure all processes have finished saving their samples before creating npz
    dist.barrier()

    if sample_speed_benchmark:
        torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Aggregate timing across all ranks
        elapsed_tensor = torch.tensor([elapsed_time], device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        max_elapsed = elapsed_tensor.item()

        if rank == 0:
            captions_per_sec = benchmark_captions / max_elapsed
            print(f"\n{'=' * 60}")
            print("Speed Benchmark Results:")
            print(f"  Warmup batches: {warmup_batches} (skipped from timing)")
            print(f"  Benchmarked captions: {benchmark_captions}")
            print(f"  Total time: {max_elapsed:.2f} seconds")
            print(f"  Throughput: {captions_per_sec:.2f} captions/sec")
            print(
                f"  Per-GPU throughput: {captions_per_sec / world_size:.2f} captions/sec"
            )
            print(f"{'=' * 60}")
    else:
        if rank == 0:
            print("Done.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
