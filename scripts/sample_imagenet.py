import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
from types import SimpleNamespace
import torch
import torch.distributed as dist
import os
import time
from utils.logger import setup_logger
from utils.train_utils import create_dataloader, get_pretrained_tokenizer
from utils.eval_utils import load_refs_from_wds, compute_metrics
from modeling.generator import BAR


def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


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
            print(f"Saving .png samples at {sample_folder_dir}")
    else:
        if rank == 0:
            print("Speed benchmark mode: skipping image saving and npz creation")
            print("GPU warmup: first 10 batches will be excluded from timing")
    dist.barrier()

    n = per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    output_dir = config.experiment.output_dir
    accelerator = SimpleNamespace(
        num_processes=world_size,
        process_index=rank,
        device=torch.device(f"cuda:{local_rank}"),
    )

    logger = setup_logger(
        name="SAMPLE",
        log_level="INFO",
        output_file=f"{output_dir}/log{accelerator.process_index}.txt",
        use_accelerate=False,
    )

    _, eval_dataloader = create_dataloader(config, logger, accelerator)
    total = 0

    # Benchmark variables
    warmup_batches = 10 if sample_speed_benchmark else 0
    start_time = None
    benchmark_captions = 0
    preds = {}

    for batch_idx, batch in enumerate(eval_dataloader):
        # Start timing after warmup batches
        if sample_speed_benchmark and batch_idx == warmup_batches:
            torch.cuda.synchronize()
            start_time = time.time()

        # Generate tokens
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

        generated_tokens = generator.generate(
            condition=conditions,
            guidance_scale=config.model.generator.guidance_scale,
            randomize_temperature=config.model.generator.mbm_head.randomize_temperature,
            kv_cache=True,
            tokens_allocation=tokens_allocation,
        )

        generated_captions = tokenizer.decode_tokens(generated_tokens)

        preds.update(
            {image_id: [cap] for image_id, cap in zip(image_ids, generated_captions)}
        )

        if not sample_speed_benchmark:
            for i, sample in enumerate(generated_captions):
                index = i * dist.get_world_size() + rank + total
                filename = f"{index:06d}.txt"
                with open(f"{sample_folder_dir}/{filename}", "w") as f:
                    f.write(sample)

        # Count captions after warmup for benchmark
        if sample_speed_benchmark and batch_idx >= warmup_batches:
            benchmark_captions += global_batch_size

        total += global_batch_size

    refs = load_refs_from_wds(config.dataset.params.eval_shards_path_or_url)
    metrics = compute_metrics(preds, refs)
    logger.info("Metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

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
