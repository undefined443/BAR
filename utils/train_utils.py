"""Training utils."""

import json
import os
import time
import math
from pathlib import Path
import wandb

from data import SimpleImageDataset, CachedTokensFolder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel
from modeling.generator import BAR
from evaluator import VQGANEvaluator

import torchvision.transforms.functional as TVF

from modeling.tokenizer import BAR_FSQ


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.

    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_pretrained_tokenizer(config):
    """Load pretrained tokenizer.

    Args:
        config: Config object. If config.tokenizer_config is specified,
                the model.vq_model section should already be merged into config.

    Returns:
        Loaded tokenizer model in eval mode with gradients disabled.
    """
    model = BAR_FSQ(config)
    model.eval()
    model.requires_grad_(False)
    return model


def create_model_and_loss_module(config, logger, accelerator):
    """Creates BAR generator model and loss module."""
    logger.info("Creating model and loss module.")
    model_cls = BAR
    loss_cls = None  # but we will not use
    model = model_cls(config)
    print(model)

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(
            model.parameters(), decay=0.999, model_cls=model_cls, config=config
        )

        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "ema_model"), model_cls=model_cls, config=config
            )
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Create loss module along with discrminator.
    loss_module = loss_cls(config=config) if loss_cls is not None else nn.Identity()

    # Print model size
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("=" * 80)
        logger.info(f"Model: {model_cls.__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        logger.info("=" * 80)

    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module):
    """Creates optimizer for BAR_FSQ tokenizer and discriminator.

    Rules:
    - All norm layers, embedding layers, and bias terms have weight_decay=0
    - Encoder parameters use lr * encoder_lr_mult
    - Other parameters use base lr
    """
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate
    discriminator_learning_rate = optimizer_config.get(
        "discriminator_learning_rate", 1e-4
    )

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    elif optimizer_type == "adamw_8bit":
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Check if param should have weight decay 0: norm layers, embedding layers, bias terms
    def no_weight_decay(n, p):
        return (
            p.ndim < 2  # Bias terms and 1D params
            or "norm" in n.lower()
            or "ln" in n.lower()  # Norm layers
            or "bias" in n  # Bias terms
            or "embedding" in n.lower()
            or "embed" in n.lower()
        )  # Embedding layers

    named_parameters = list(model.named_parameters())

    # Separate encoder and non-encoder parameters
    encoder_params_no_decay = [
        p
        for n, p in named_parameters
        if "encoder" in n and no_weight_decay(n, p) and p.requires_grad
    ]
    encoder_params_with_decay = [
        p
        for n, p in named_parameters
        if "encoder" in n and not no_weight_decay(n, p) and p.requires_grad
    ]

    other_params_no_decay = [
        p
        for n, p in named_parameters
        if "encoder" not in n and no_weight_decay(n, p) and p.requires_grad
    ]
    other_params_with_decay = [
        p
        for n, p in named_parameters
        if "encoder" not in n and not no_weight_decay(n, p) and p.requires_grad
    ]

    encoder_lr_mult = config.model.vq_model.get("encoder_lr_mult", 1.0)

    param_groups = [
        {
            "params": encoder_params_no_decay,
            "weight_decay": 0.0,
            "lr": learning_rate * encoder_lr_mult,
        },
        {
            "params": encoder_params_with_decay,
            "weight_decay": optimizer_config.weight_decay,
            "lr": learning_rate * encoder_lr_mult,
        },
        {"params": other_params_no_decay, "weight_decay": 0.0},
        {
            "params": other_params_with_decay,
            "weight_decay": optimizer_config.weight_decay,
        },
    ]

    # Log parameter group info
    logger.info(f"Encoder params (no decay): {len(encoder_params_no_decay)}")
    logger.info(f"Encoder params (with decay): {len(encoder_params_with_decay)}")
    logger.info(f"Other params (no decay): {len(other_params_no_decay)}")
    logger.info(f"Other params (with decay): {len(other_params_with_decay)}")
    logger.info(f"Encoder LR multiplier: {encoder_lr_mult}")

    optimizer = optimizer_cls(
        param_groups,
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
    )

    # Discriminator optimizer (only create if discriminator is enabled)
    if hasattr(loss_module, "use_discriminator") and not loss_module.use_discriminator:
        logger.info(
            "Discriminator is disabled (discriminator_weight = 0), skipping discriminator optimizer creation."
        )
        discriminator_optimizer = None
    else:
        discriminator_named_parameters = list(loss_module.named_parameters())
        discriminator_params_no_decay = [
            p
            for n, p in discriminator_named_parameters
            if no_weight_decay(n, p) and p.requires_grad
        ]
        discriminator_params_with_decay = [
            p
            for n, p in discriminator_named_parameters
            if not no_weight_decay(n, p) and p.requires_grad
        ]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_params_no_decay, "weight_decay": 0.0},
                {
                    "params": discriminator_params_with_decay,
                    "weight_decay": optimizer_config.weight_decay,
                },
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
        )

    return optimizer, discriminator_optimizer


def create_lr_scheduler(
    config, logger, accelerator, optimizer, discriminator_optimizer=None
):
    """Creates learning rate scheduler for BAR_FSQ tokenizer and discriminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps
        * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=(
                config.training.max_train_steps - config.losses.discriminator_start
            )
            * accelerator.num_processes,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps
            * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None
    return lr_scheduler, discriminator_lr_scheduler


def create_dataloader(config, logger, accelerator):
    """Creates data loader for training and testing."""
    logger.info("Creating dataloaders.")
    total_batch_size_without_accum = (
        config.training.per_gpu_batch_size * accelerator.num_processes
    )
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    dataset = SimpleImageDataset(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=total_batch_size_without_accum,
        num_workers_per_gpu=dataset_config.num_workers_per_gpu,
        resize_shorter_edge=preproc_config.resize_shorter_edge,
        crop_size=preproc_config.crop_size,
        random_crop=preproc_config.random_crop,
        random_flip=preproc_config.random_flip,
    )
    train_dataloader, eval_dataloader = (
        dataset.train_dataloader,
        dataset.eval_dataloader,
    )
    # Mark that this dataloader is WebDataset-based and handles distributed training internally
    train_dataloader.is_webdataset = True

    if dataset_config.get("pretokenization", ""):
        pretok_path = dataset_config.pretokenization
        logger.info(f"Using pretokenized dataset: {pretok_path}")

        # Only NPZ format is supported: directory with train/ subdirectory containing class folders
        if not (
            os.path.isdir(pretok_path)
            and os.path.isdir(os.path.join(pretok_path, "train"))
        ):
            raise ValueError(
                f"Invalid pretokenization path: {pretok_path}\n"
                f"Expected NPZ format: a directory containing a 'train/' subdirectory with class folders.\n"
                f'Please run pretokenization first or set dataset.params.pretokenization="" to use online tokenization.'
            )

        logger.info("Detected NPZ format")

        # Check metadata
        metadata_path = os.path.join(pretok_path, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"metadata.json not found at {metadata_path}")
        else:
            import json

            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                logger.info(f"NPZ metadata: {metadata}")

        # Create dataset
        npz_dataset = CachedTokensFolder(root=os.path.join(pretok_path, "train"))
        logger.info(f"NPZ dataset: {len(npz_dataset)} samples")

        # Create dataloader (accelerator.prepare() will handle distributed sampling)
        train_dataloader = DataLoader(
            npz_dataset,
            batch_size=config.training.per_gpu_batch_size,
            shuffle=True,
            num_workers=dataset_config.num_workers_per_gpu,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
            if dataset_config.num_workers_per_gpu > 0
            else False,
        )
        train_dataloader.num_batches = math.ceil(
            config.experiment.max_train_examples / total_batch_size_without_accum
        )
        logger.info(f"Batches per epoch: {train_dataloader.num_batches}")

    return train_dataloader, eval_dataloader


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    evaluator = VQGANEvaluator(
        device=accelerator.device,
        enable_rfid=True,
        enable_inception_score=True,
        enable_psnr_score=True,
        enable_ssim_score=True,
        enable_lpips_score=True,
        enable_mse_error=True,
        enable_mae_error=True,
    )
    return evaluator


def auto_resume(
    config, logger, accelerator, ema_model, num_update_steps_per_epoch, strict=True
):
    """Auto resuming the training from checkpoint-latest."""
    global_step = 0
    first_epoch = 0
    wandb_run_id = None
    # If resuming training.
    if config.experiment.resume:
        accelerator.wait_for_everyone()
        checkpoint_path = Path(config.experiment.output_dir) / "checkpoint-latest"

        if checkpoint_path.exists():
            logger.info(f"Found checkpoint at {checkpoint_path}")
            global_step, wandb_run_id = load_checkpoint(
                checkpoint_path, accelerator, logger=logger, strict=strict
            )
            logger.info(f"Resuming training from global_step={global_step}")
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resuming from epoch {first_epoch}")
        else:
            logger.info("No checkpoint found. Training from scratch.")
    return global_step, first_epoch, wandb_run_id


def generator_train_one_epoch(
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
    wandb_run_id=None,
):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    loss_average_meter = AverageMeter()
    end = time.time()
    model.train()

    def get_rar_random_ratio(config, cur_step):
        # set both to -1 for a raster training
        randomness_anneal_start = config.model.generator.randomness_anneal_start
        randomness_anneal_end = config.model.generator.randomness_anneal_end
        if cur_step < randomness_anneal_start:
            return 1.0
        elif cur_step > randomness_anneal_end:
            return 0.0
        else:
            return 1.0 - (cur_step - randomness_anneal_start) / (
                randomness_anneal_end - randomness_anneal_start
            )

    for batch in train_dataloader:
        model.train()
        if config.dataset.params.get("pretokenization", ""):
            conditions, input_tokens = batch
            input_tokens = input_tokens.to(
                accelerator.device,
                memory_format=torch.contiguous_format,
                non_blocking=True,
            )
            conditions = conditions.to(
                accelerator.device,
                memory_format=torch.contiguous_format,
                non_blocking=True,
            )
        elif "caption" in batch:
            captions = batch["caption"]
            images = batch["image"].to(
                accelerator.device,
                memory_format=torch.contiguous_format,
                non_blocking=True,
            )
            # Encode captions on the flight.
            with torch.no_grad():
                if tokenizer_encode_fn is not None:
                    input_tokens, conditions = tokenizer_encode_fn(captions, images)
                else:
                    # Fallback: call tokenizer.encode directly
                    input_tokens, conditions = tokenizer.encode(captions, images)
                input_tokens = input_tokens.reshape(len(captions), -1)
        else:
            raise NotImplementedError

        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)
        # Compute random_ratio outside compiled region to avoid recompilation
        random_ratio = get_rar_random_ratio(config, global_step)
        unwrap_model.set_random_ratio(
            random_ratio
        )  # Keep for backward compat and inference

        # Sample orders outside compiled region to avoid recompilation/slowdown
        # when random_ratio changes during training
        orders = unwrap_model.sample_orders(input_tokens, random_ratio=random_ratio)

        with accelerator.accumulate([model]):
            condition = unwrap_model.preprocess_condition(
                conditions, cond_drop_prob=config.model.generator.class_label_dropout
            )

            # Pass pre-sampled orders to avoid torch.compile seeing different patterns
            gen_loss = model(input_tokens, condition, orders=orders)
            loss_dict = {"loss": gen_loss}

            # Gather the losses across all processes for logging.
            gen_logs = {}
            for k, v in loss_dict.items():
                gen_logs["train/" + k] = accelerator.gather(v).mean().item()
            accelerator.backward(gen_loss)

            if accelerator.sync_gradients:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.training.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

        loss_average_meter.update(gen_logs["train/loss"])
        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps
                    * config.training.per_gpu_batch_size
                    / batch_time_meter.val
                )
                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Loss: {loss_average_meter.avg:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                    "train/avg_mlm_loss": loss_average_meter.avg,
                }
                logs.update(gen_logs)
                logs.update({"random_ratio": unwrap_model.random_ratio})
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()
                loss_average_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_checkpoint(
                    model,
                    config.experiment.output_dir,
                    accelerator,
                    global_step + 1,
                    logger=logger,
                    wandb_run_id=wandb_run_id,
                )
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate captions.
            if (
                global_step + 1
            ) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Generate captions with non-EMA model
                generate_captions(
                    model,
                    tokenizer,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    eval_dataloader=eval_dataloader,
                    config=config,
                    model_type="Non-EMA",
                )

                # Generate captions with EMA model
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                    generate_captions(
                        model,
                        tokenizer,
                        accelerator,
                        global_step + 1,
                        config.experiment.output_dir,
                        logger=logger,
                        eval_dataloader=eval_dataloader,
                        config=config,
                        model_type="EMA",
                    )

                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            global_step += 1
            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break
    return global_step


@torch.no_grad()
def generate_captions(
    model,
    tokenizer,
    accelerator,
    global_step,
    output_dir,
    logger,
    eval_dataloader,
    config=None,
    model_type="",
):
    model_suffix = f" ({model_type})" if model_type else ""
    logger.info(f"Generating captions{model_suffix}...")

    # Generate with random order sampling
    logger.info(f"Generating captions with random order sampling{model_suffix}...")
    generated_caption_random = sample_captions(
        model,
        tokenizer,
        eval_dataloader,
        num_samples=config.training.num_generated_captions,
        config=config,
        accelerator=accelerator,
        sample_with_random_order=True,
    )

    # Log captions.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log(
            {
                f"Train Generated (Random Order){model_suffix}": [
                    wandb.Image(item["image"], caption=item["caption"])
                    for item in generated_caption_random
                ]
            },
            step=global_step,
        )
    else:
        raise NotImplementedError("TensorBoard does not support image-caption logging")

    # Generate with raster order sampling
    logger.info(f"Generating captions with raster order sampling{model_suffix}...")
    generated_caption_raster = sample_captions(
        model,
        tokenizer,
        eval_dataloader,
        num_samples=config.training.num_generated_captions,
        config=config,
        accelerator=accelerator,
        sample_with_random_order=False,
    )

    # Log captions.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log(
            {
                f"Train Generated (Raster Order){model_suffix}": [
                    wandb.Image(item["image"], caption=item["caption"])
                    for item in generated_caption_raster
                ]
            },
            step=global_step,
        )
    else:
        raise NotImplementedError("TensorBoard does not support image-caption logging")

    return


@torch.no_grad()
@torch._dynamo.disable
def sample_captions(
    generator,
    tokenizer,
    eval_dataloader,
    num_samples: int = 10,
    config=None,
    accelerator=None,
    device=None,
    sample_with_random_order=None,
):
    generator.eval()
    tokenizer.eval()
    if device is None:
        device = accelerator.device

    if accelerator is None:
        unwrap_generator = generator
    else:
        unwrap_generator = accelerator.unwrap_model(generator)

    if sample_with_random_order is None:
        sample_with_random_order = unwrap_generator.random_ratio >= 1.0

    # Determine dtype for sampling based on mixed precision setting
    if accelerator is not None:
        if accelerator.mixed_precision == "fp16":
            dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        use_autocast = accelerator.mixed_precision != "no"
    else:
        dtype = torch.float32
        use_autocast = False

    generated_captions = []

    for batch in eval_dataloader:
        if len(generated_captions) >= num_samples:
            break
        captions = batch["caption"]
        images = batch["image"].to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )
        image_ids = batch["__key__"]
        # Encode captions on the flight.
        with torch.no_grad():
            _, conditions = tokenizer.encode(captions, images)

        condition = unwrap_generator.preprocess_condition(
            conditions, cond_drop_prob=config.model.generator.class_label_dropout
        )

        # Use autocast for FlashAttention compatibility
        with torch.autocast("cuda", dtype=dtype, enabled=use_autocast):
            generated_tokens = unwrap_generator.generate(
                condition=condition,
                guidance_scale=config.model.generator.guidance_scale,
                randomize_temperature=config.model.generator.mbm_head.randomize_temperature,
                tokens_allocation=config.model.generator.mbm_head.get(
                    "tokens_allocation", None
                ),
                sample_with_random_order=sample_with_random_order,
                kv_cache=True,
            )
            generated_caption = tokenizer.decode_tokens(generated_tokens)
            generated_captions.extend(
                {
                    "image_id": int(image_id),
                    "caption": cap,
                    "image": TVF.to_pil_image(
                        ((img.cpu().float() + 1) / 2).clamp(0, 1)
                    ),
                }
                for image_id, cap, img in zip(image_ids, generated_caption, images)
            )

    generator.train()
    return generated_captions


def save_checkpoint(
    model, output_dir, accelerator, global_step, logger, wandb_run_id=None
) -> Path:
    """Save checkpoint to 'checkpoint-latest' using atomic rename for safety.

    Saves to a temporary folder first, then atomically replaces the old checkpoint.
    This prevents data loss if saving fails due to insufficient storage or crashes.
    """
    import shutil

    output_dir = Path(output_dir)
    final_path = output_dir / "checkpoint-latest"
    temp_path = output_dir / "checkpoint-latest-new"

    # Save to temporary location first
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            temp_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        metadata = {"global_step": global_step}
        if wandb_run_id is not None:
            metadata["wandb_run_id"] = wandb_run_id
        json.dump(metadata, (temp_path / "metadata.json").open("w+"))

    accelerator.save_state(temp_path)

    # Wait for all processes to finish saving
    accelerator.wait_for_everyone()

    # Atomically replace old checkpoint with new one (only main process)
    if accelerator.is_main_process:
        if final_path.exists():
            # Remove old checkpoint after new one is successfully saved
            logger.info(f"Removing old checkpoint at {final_path}")
            shutil.rmtree(final_path)

        # Rename new checkpoint to final location
        temp_path.rename(final_path)
        logger.info(f"Saved checkpoint to {final_path} (global_step={global_step})")

    # Wait for rename to complete
    accelerator.wait_for_everyone()

    return final_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)

    with open(checkpoint_path / "metadata.json", "r") as f:
        metadata = json.load(f)
        global_step = int(metadata["global_step"])
        wandb_run_id = metadata.get("wandb_run_id", None)

    logger.info(f"Resuming at global_step {global_step}")
    if wandb_run_id:
        logger.info(f"Resuming wandb run with ID: {wandb_run_id}")
    return global_step, wandb_run_id


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)
