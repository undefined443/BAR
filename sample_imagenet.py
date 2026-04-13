from omegaconf import OmegaConf
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import os
import time
from tqdm import tqdm

from utils.train_utils import get_pretrained_tokenizer
from modeling.generator import BAR


def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main():
    config = get_config_cli()
    num_fid_samples = config.experiment.get("num_fid_samples", 50000)
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

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    assert num_fid_samples % global_batch_size == 0
    if rank == 0:
        print(f"Total number of images that will be sampled: {num_fid_samples}")

    samples_needed_this_gpu = int(num_fid_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, (
        "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    )
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    # Use all classes (balanced across num_fid_samples)
    class_list = list(range(config.model.generator.condition_num_classes))
    num_classes = len(class_list)
    all_classes = class_list * (num_fid_samples // num_classes + 1)
    all_classes = all_classes[:num_fid_samples]  # Trim to exact number

    subset_len = len(all_classes) // world_size
    all_classes = np.array(
        all_classes[rank * subset_len : (rank + 1) * subset_len], dtype=np.int64
    )
    cur_idx = 0

    # Benchmark variables
    warmup_batches = 10 if sample_speed_benchmark else 0
    start_time = None
    benchmark_images = 0

    for batch_idx in pbar:
        # Start timing after warmup batches
        if sample_speed_benchmark and batch_idx == warmup_batches:
            torch.cuda.synchronize()
            start_time = time.time()

        y = torch.from_numpy(all_classes[cur_idx * n : (cur_idx + 1) * n]).to(device)
        cur_idx += 1

        # Generate tokens
        tokens_allocation = config.model.generator.mbm_head.get(
            "tokens_allocation", None
        )

        generated_tokens = generator.generate(
            condition=y.long(),
            guidance_scale=config.model.generator.guidance_scale,
            randomize_temperature=config.model.generator.mbm_head.randomize_temperature,
            kv_cache=True,
            tokens_allocation=tokens_allocation,
        )

        generated_image = tokenizer.decode_tokens(generated_tokens)
        # shift from [-1, 1] to [0, 1]
        generated_image = (generated_image + 1.0) / 2.0

        if not sample_speed_benchmark:
            samples = torch.clamp(generated_image, 0.0, 1.0)
            samples = (
                (samples * 255.0)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                filename = f"{index:06d}.png"
                Image.fromarray(sample).save(f"{sample_folder_dir}/{filename}")

        # Count images after warmup for benchmark
        if sample_speed_benchmark and batch_idx >= warmup_batches:
            benchmark_images += global_batch_size

        total += global_batch_size

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
            images_per_sec = benchmark_images / max_elapsed
            print(f"\n{'=' * 60}")
            print("Speed Benchmark Results:")
            print(f"  Warmup batches: {warmup_batches} (skipped from timing)")
            print(f"  Benchmarked images: {benchmark_images}")
            print(f"  Total time: {max_elapsed:.2f} seconds")
            print(f"  Throughput: {images_per_sec:.2f} images/sec")
            print(f"  Per-GPU throughput: {images_per_sec / world_size:.2f} images/sec")
            print(f"{'=' * 60}")
    else:
        if rank == 0:
            create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
            print("Done.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
