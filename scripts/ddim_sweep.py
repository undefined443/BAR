"""Benchmark DDIM sampling quality across different num_steps values.

Launch with accelerate:
    uv run accelerate launch --num_machines=1 --num_processes=<N> \\
        scripts/ddim_sweep.py \\
        config=<config.yaml> \\
        experiment.wandb_run_id=<wandb_run_id> \\
        [experiment.global_step=<step>] \\
        [bench.steps=[1,5,10,20,50,100]] \\
        [bench.project=<wandb_project>] \\
        [bench.no_wandb=true]

The checkpoint is downloaded from a wandb artifact. experiment.global_step
selects a specific checkpoint.

The model is loaded once. Each batch is encoded once, then generated for
all num_steps values before moving to the next batch.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import dotenv
import torch
import wandb
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.generator import BAR
from utils.eval_utils import compute_metrics, load_refs_from_wds
from utils.logger import setup_logger
from utils.train_utils import create_dataloader, get_pretrained_tokenizer


STEPS_DEFAULT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100, 150, 200]


def get_config_cli():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def print_summary(steps, all_metrics):
    orders, metric_keys = [], []
    for m in all_metrics.values():
        if m:
            orders = list(m.keys())
            metric_keys = list(next(iter(m.values())).keys())
            break

    if not orders:
        print("No metrics found.")
        return

    for order in orders:
        print(f"\n=== {order} ===")
        col_w = 10
        header = f"{'steps':>6}  " + "  ".join(f"{k:>{col_w}}" for k in metric_keys)
        print(header)
        print("-" * len(header))
        for num_steps in steps:
            row = all_metrics.get(num_steps, {}).get(order)
            if row:
                vals = "  ".join(
                    f"{row.get(k, float('nan')):>{col_w}.4f}" for k in metric_keys
                )
                print(f"{num_steps:>6}  {vals}")
            else:
                print(f"{num_steps:>6}  (missing)")


def _find_checkpoint_artifact(api, entity, project, wandb_run_id, global_step):
    artifact_name = f"checkpoint-{wandb_run_id}"
    full_name = f"{entity}/{project}/{artifact_name}"
    try:
        versions = api.artifacts("model", full_name)
        for v in versions:
            if v.metadata.get("global_step") == global_step:
                return v
    except wandb.errors.CommError:
        pass

    raise RuntimeError(
        f"Checkpoint artifact not found for run {wandb_run_id}, step {global_step}"
    )


def download_checkpoint(entity, project, wandb_run_id, global_step):
    """Download a checkpoint artifact from wandb, using wandb's local cache.

    Returns (checkpoint_path, checkpoint_dir).
    """
    if not all([entity, project, wandb_run_id, global_step]):
        raise ValueError(
            "entity, project, wandb_run_id, and global_step must all be provided to download checkpoint"
        )
    api = wandb.Api()
    artifact = _find_checkpoint_artifact(
        api, entity, project, wandb_run_id, global_step
    )

    print(f"Downloading artifact {artifact.name} (cached in ~/.cache/wandb)")
    checkpoint_dir = Path(artifact.download())
    checkpoint_path = checkpoint_dir / "unwrapped_model" / "pytorch_model.bin"
    return checkpoint_path


def log_to_wandb(
    steps, all_metrics, wandb_run_id, global_step, project, entity, experiment_name
):
    orders = []
    for m in all_metrics.values():
        if m:
            orders = list(m.keys())
            break

    if not orders:
        return

    run = wandb.init(
        project=project,
        entity=entity,
        name=f"ddim_sweep/{experiment_name}-{global_step}",
        config={
            "training_run_id": wandb_run_id,
            "global_step": global_step,
            "experiment_name": experiment_name,
            "steps": steps,
        },
        job_type="bench_diffusion_steps",
    )

    api = wandb.Api()
    artifact = _find_checkpoint_artifact(
        api, entity, project, wandb_run_id, global_step
    )
    run.use_artifact(artifact)

    order_postfix = {
        order: ("raster" if "Raster" in order else "random") for order in orders
    }
    for num_steps in steps:
        log_dict = {}
        for order, metrics in (all_metrics.get(num_steps) or {}).items():
            postfix = order_postfix[order]
            for k, v in metrics.items():
                log_dict[f"{k}/{postfix}"] = v
        wandb.log(log_dict, step=num_steps)

    run.finish()


def main():
    dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    config = get_config_cli()
    project = config.experiment.get("project", None)
    enable_wandb = config.training.get("enable_wandb", False)
    bench_cfg = config.get("bench", {})
    steps = list(bench_cfg.get("steps", STEPS_DEFAULT))
    entity = os.environ.get("WANDB_ENTITY")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    accelerator = Accelerator(
        mixed_precision=config.training.get("mixed_precision", "no")
    )
    set_seed(config.experiment.get("random_seed", 42), device_specific=True)

    wandb_run_id = config.experiment.get("wandb_run_id")
    global_step = config.experiment.get("global_step")
    with accelerator.main_process_first():
        checkpoint_path = download_checkpoint(
            entity, project, wandb_run_id, global_step
        )

    logger = setup_logger(name="BENCH", log_level="INFO", use_accelerate=False)

    tokenizer = get_pretrained_tokenizer(config)
    tokenizer.to(accelerator.device)

    generator = BAR(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    generator.load_state_dict(cleaned_state_dict)
    generator.eval()
    generator.requires_grad_(False)
    generator.to(accelerator.device)

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in generator.parameters())
        print(f"Generator parameters: {total_params:,}")
        print(f"Training run: {wandb_run_id}")
        print(f"Global step: {global_step}")
        print(f"Sweep steps: {steps}")

    per_proc_batch_size = config.experiment.get("per_proc_batch_size", 125)
    global_batch_size = per_proc_batch_size * accelerator.num_processes
    total_batches = 5000 // global_batch_size

    _, eval_dataloader = create_dataloader(config, logger, accelerator)

    refs = (
        load_refs_from_wds(config.dataset.params.eval_shards_path_or_url)
        if accelerator.is_main_process
        else None
    )

    tokens_allocation = config.model.generator.mbm_head.get("tokens_allocation", None)
    base_generate_kwargs = dict(
        guidance_scale=config.model.generator.guidance_scale,
        randomize_temperature=config.model.generator.mbm_head.randomize_temperature,
        kv_cache=True,
        tokens_allocation=tokens_allocation,
    )

    preds_raster = defaultdict(dict)
    preds_random = defaultdict(dict)

    progress = tqdm(
        eval_dataloader,
        total=total_batches,
        desc="Sampling",
        unit="batch",
        disable=not accelerator.is_main_process,
    )
    for batch_idx, batch in enumerate(progress):
        if batch_idx >= total_batches:
            break

        captions = batch["caption"]
        images = batch["image"].to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )
        image_ids = [int(k) for k in batch["__key__"]]

        with torch.no_grad():
            _, conditions = tokenizer.encode(captions, images)

        for num_steps in steps:
            kwargs = {
                **base_generate_kwargs,
                "condition": conditions,
                "num_steps": num_steps,
            }
            raster_caps = tokenizer.decode_tokens(
                generator.generate(**kwargs, sample_with_random_order=False)
            )
            random_caps = tokenizer.decode_tokens(
                generator.generate(**kwargs, sample_with_random_order=True)
            )
            preds_raster[num_steps].update(
                {iid: [cap] for iid, cap in zip(image_ids, raster_caps)}
            )
            preds_random[num_steps].update(
                {iid: [cap] for iid, cap in zip(image_ids, random_caps)}
            )

    all_metrics = {}
    for num_steps in steps:
        all_raster = [None] * accelerator.num_processes
        all_random = [None] * accelerator.num_processes
        dist.all_gather_object(all_raster, dict(preds_raster[num_steps]))
        dist.all_gather_object(all_random, dict(preds_random[num_steps]))

        if accelerator.is_main_process:
            merged_raster = {k: v for d in all_raster for k, v in d.items()}
            merged_random = {k: v for d in all_random for k, v in d.items()}
            step_metrics = {}
            for order, merged_preds in [
                ("Raster Order", merged_raster),
                ("Random Order", merged_random),
            ]:
                metrics = compute_metrics(merged_preds, refs)
                logger.info(
                    f"num_steps={num_steps} Metrics ({order}): "
                    + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                )
                step_metrics[order] = metrics
            all_metrics[num_steps] = step_metrics

    if accelerator.is_main_process and enable_wandb:
        log_to_wandb(
            steps,
            all_metrics,
            wandb_run_id,
            global_step,
            project,
            entity,
            config.experiment.name,
        )


if __name__ == "__main__":
    main()
