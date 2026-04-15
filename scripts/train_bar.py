"""Training script for image generator."""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import os

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    get_config,
    create_model_and_loss_module,
    get_pretrained_tokenizer,
    create_optimizer,
    create_lr_scheduler,
    create_dataloader,
    auto_resume,
    save_checkpoint,
    generator_train_one_epoch,
)


def main():
    # Set up the workspace, in case we do not have Internet access, etc. In this case
    # we would like to use cached pretrain weights.
    workspace = os.environ.get("WORKSPACE", "")
    torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()

    # If tokenizer_config is provided, merge the model.vq_model section from it
    if config.get("tokenizer_config", None):
        tokenizer_config_path = config.tokenizer_config
        print(f"Loading tokenizer config from: {tokenizer_config_path}")
        tokenizer_config = OmegaConf.load(tokenizer_config_path)

        # Copy the vq_model configuration from tokenizer config
        if "model" in tokenizer_config and "vq_model" in tokenizer_config.model:
            print("Copying model.vq_model configuration from tokenizer config")
            if "model" not in config:
                config.model = OmegaConf.create({})
            # Merge instead of replace to allow overrides in generator config
            config.model.vq_model = OmegaConf.merge(
                tokenizer_config.model.vq_model, config.model.get("vq_model", {})
            )
        else:
            print("Warning: tokenizer config does not contain model.vq_model section")
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(
        name="GEN",
        log_level="INFO",
        output_file=f"{output_dir}/log{accelerator.process_index}.txt",
    )

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    tokenizer = get_pretrained_tokenizer(config)
    tokenizer.to(accelerator.device)

    # Set tokenizer mode once based on config (for torch.compile compatibility)
    online_tokenization_with_noises = config.training.get(
        "online_tokenization_with_noises", False
    )
    if online_tokenization_with_noises:
        tokenizer.train()  # Enable noising during tokenization
        logger.info("Online tokenization with noises ENABLED (tokenizer in train mode)")
    else:
        tokenizer.eval()  # Disable noising during tokenization
        logger.info("Online tokenization with noises DISABLED (tokenizer in eval mode)")

    tokenizer_encode_fn = None
    if not config.dataset.params.get("pretokenization", ""):

        def _tokenizer_encode(texts, images):
            # Mode is already set above; don't change it here to avoid torch.compile issues
            token_ids, embeddings = tokenizer.encode(texts, images)
            return token_ids, embeddings

        tokenizer_encode_fn = _tokenizer_encode

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator
    )

    optimizer, _ = create_optimizer(config, logger, model, loss_module)

    lr_scheduler, _ = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer=None
    )

    train_dataloader, eval_dataloader = create_dataloader(config, logger, accelerator)

    eval_dataloader = None

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    # WebDataset-based dataloaders are already aware of distributed training, so we don't need to prepare them.
    # Only prepare dataloaders for JSONL and NPZ pretokenization formats.
    is_webdataset = getattr(train_dataloader, "is_webdataset", False)
    if config.dataset.params.get("pretokenization", "") and not is_webdataset:
        # JSONL or NPZ pretokenization: needs accelerator.prepare() for distributed training
        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader
        )
    else:
        # Online tokenization (WebDataset) or pretokenized WebDataset: already handles distributed training
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )

    # Compile model after accelerator.prepare()
    if config.training.get("compile_model", False):
        logger.info("Compiling generator with torch.compile")
        model = torch.compile(model)
        if tokenizer_encode_fn is not None:
            logger.info("Compiling tokenizer encode() with torch.compile")
            tokenizer_encode_fn = torch.compile(tokenizer_encode_fn)
        # loss_module is nn.Identity() for BAR, no need to compile
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    total_batch_size_without_accum = (
        config.training.per_gpu_batch_size * accelerator.num_processes
    )
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        num_batches / config.training.gradient_accumulation_steps
    )

    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(
        config.training.max_train_steps / num_update_steps_per_epoch
    )

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(
        f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"  Instantaneous batch size per gpu = {config.training.per_gpu_batch_size}"
    )
    logger.info(
        f"""  Total train batch size (w. parallel, distributed & accumulation) = {
            (
                config.training.per_gpu_batch_size
                * accelerator.num_processes
                * config.training.gradient_accumulation_steps
            )
        }"""
    )
    global_step = 0
    first_epoch = 0

    global_step, first_epoch, wandb_run_id = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch
    )

    # Initialize trackers after auto_resume to support wandb resume
    if accelerator.is_main_process:
        init_kwargs = {"wandb": {"name": config.experiment.name}}
        # Resume wandb run if we have a run ID
        if wandb_run_id is not None:
            init_kwargs["wandb"]["id"] = wandb_run_id
            init_kwargs["wandb"]["resume"] = "allow"

        accelerator.init_trackers(
            project_name=config.experiment.project, init_kwargs=init_kwargs
        )

        # Get the wandb run ID for saving in checkpoints
        if config.training.enable_wandb:
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_run_id = wandb_tracker.run.id
            logger.info(f"Wandb run ID: {wandb_run_id}")

        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs - 1} started.")
        global_step = generator_train_one_epoch(
            config,
            logger,
            accelerator,
            model,
            ema_model,
            loss_module,
            optimizer,
            lr_scheduler,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            tokenizer_encode_fn,
            global_step,
            wandb_run_id=wandb_run_id,
        )
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(
        model,
        output_dir,
        accelerator,
        global_step,
        logger=logger,
        wandb_run_id=wandb_run_id,
    )
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
