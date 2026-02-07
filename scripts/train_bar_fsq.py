"""Training script for image tokenizer."""
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
    get_config, create_model_and_loss_module,
    create_optimizer, create_lr_scheduler, create_dataloader,
    create_evaluator, auto_resume, save_checkpoint, 
    train_one_epoch)


def main():
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()

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
        name="Tok",
        log_level="INFO",
        output_file=f"{output_dir}/log{accelerator.process_index}.txt"
    )

    # Enable TF32 on Ampere GPUs for faster training
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for faster training on Ampere GPUs")

    # Enable cudnn benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type="tokenizer")

    optimizer, discriminator_optimizer = create_optimizer(config, logger, model, loss_module)

    lr_scheduler, discriminator_lr_scheduler = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer)

    train_dataloader, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    if discriminator_optimizer is not None:
        model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler = accelerator.prepare(
            model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler
        )
    else:
        # Discriminator is disabled, don't prepare discriminator components
        model, loss_module, optimizer, lr_scheduler = accelerator.prepare(
            model, loss_module, optimizer, lr_scheduler
        )
        discriminator_optimizer = None
        discriminator_lr_scheduler = None

    # Compile model after accelerator.prepare()
    if config.training.get("compile_model", False):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)
        loss_module = torch.compile(loss_module)


    if config.training.use_ema:
        ema_model.to(accelerator.device)
    
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Multiply by 2 because we alternate between generator and discriminator training steps
    # Each "epoch" will process the data twice: once for generator, once for discriminator
    num_train_epochs = num_train_epochs * 2
    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Num epochs (for bookkeeping) = {num_train_epochs}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = {config.training.per_gpu_batch_size}")
    total_batch_size = (
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps
    )
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    global_step = 0
    first_epoch = 0

    global_step, first_epoch, wandb_run_id = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch,
        strict= config.experiment.get("strict_loading", True))

    # Initialize trackers after auto_resume to support wandb resume
    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": config.experiment.name
            }
        }
        # Resume wandb run if we have a run ID
        if wandb_run_id is not None:
            init_kwargs["wandb"]["id"] = wandb_run_id
            init_kwargs["wandb"]["resume"] = "allow"

        accelerator.init_trackers(
            project_name=config.experiment.project,
            init_kwargs=init_kwargs)

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
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer, discriminator_optimizer,
                            lr_scheduler, discriminator_lr_scheduler,
                            train_dataloader, eval_dataloader,
                            evaluator,
                            global_step,
                            wandb_run_id=wandb_run_id)
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger, wandb_run_id=wandb_run_id)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()