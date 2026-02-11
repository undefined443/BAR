# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BAR (masked Bit AutoRegressive modeling) is a discrete visual generation framework for autoregressive image generation. It uses a two-stage pipeline: (1) a discrete tokenizer (BAR-FSQ) that encodes images into binary tokens via Finite Scalar Quantization, and (2) an autoregressive transformer with a Masked Bit Modeling (MBM) head that generates tokens through progressive bit-wise unmasking.

## Commands

### Environment
```bash
uv sync                    # install dependencies
uv run <script>            # run any script
```

### Training
```bash
# Train tokenizer (BAR-FSQ) - uses accelerate for multi-GPU
WANDB_MODE=offline WORKSPACE=/path/to/workspace \
accelerate launch --num_machines=1 --num_processes=8 \
    --machine_rank=0 --main_process_ip=127.0.0.1 \
    --main_process_port=9999 --same_network \
    scripts/train_bar_fsq.py \
    config=configs/tokenizer/bar_fsq_16bits.yaml \
    dataset.params.train_shards_path_or_url=/path/to/train.tar \
    dataset.params.eval_shards_path_or_url=/path/to/val.tar

# Train generator (BAR) - uses accelerate for multi-GPU
WANDB_MODE=offline WORKSPACE=/path/to/workspace \
accelerate launch --num_machines=1 --num_processes=8 \
    --machine_rank=0 --main_process_ip=127.0.0.1 \
    --main_process_port=9999 --same_network \
    scripts/train_bar.py \
    config=configs/generator/bar_b.yaml \
    dataset.params.pretokenization=./pretokenized_npz
```

### Sampling & Evaluation
```bash
# Generate ImageNet samples
torchrun --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:9999 \
    sample_imagenet.py \
    config=configs/generator/bar_b.yaml \
    experiment.output_dir="bar_b" \
    experiment.generator_checkpoint=assets/generator/bar_b.bin \
    experiment.tokenizer_checkpoint=assets/tokenizer/bar_fsq_16bits_ft.bin \
    model.generator.guidance_scale=5.0

# Evaluate FID (requires guided-diffusion repo cloned separately)
python3 guided-diffusion/evaluations/evaluator.py \
    VIRTUAL_imagenet256_labeled.npz output_dir.npz
```

### Pretokenization (optional, speeds up generator training)
```bash
torchrun --nproc_per_node=8 scripts/pretokenization.py \
    --img_size 256 --batch_size 32 \
    --vae_config_path configs/tokenizer/bar_fsq_16bits.yaml \
    --vae_path assets/tokenizer/bar_fsq_16bits.bin \
    --data_path /path/to/imagenet \
    --cached_path ./pretokenized_npz
```

## Architecture

### Two-Stage Pipeline

**Stage 1 - Tokenizer (`modeling/tokenizer.py: BAR_FSQ`):**
- SigLIP2-based encoder/decoder with FSQ quantizer (`modeling/quantizer.py: FSQ`)
- Binary quantization: each token is `token_size` bits (e.g., 16 bits = codebook size 65536)
- Trained with perceptual losses (LPIPS, CLIP, gram) + GAN discriminator (`modeling/modules/discriminator_dino.py`)

**Stage 2 - Generator (`modeling/generator.py: BAR`):**
- Autoregressive transformer with causal attention and RoPE
- Supports token patchification (`patch_size` > 1) to reduce sequence length
- Class-conditional via label embedding with configurable dropout
- MBM head (`modeling/mbm_head.py: MaskBitModelingHead`) replaces the linear classification head:
  - Generates tokens via iterative bit unmasking with configurable schedule (`tokens_allocation`)
  - Uses adaLN-modulated ResBlocks for conditional generation
  - Memory scales O(log2 C) instead of O(C) with codebook size C

### Config System
- OmegaConf YAML configs in `configs/` with CLI overrides (dot-notation, e.g., `model.generator.guidance_scale=5.0`)
- Generator configs reference tokenizer checkpoints via `experiment.tokenizer_checkpoint`
- Key generation hyperparams: `guidance_scale`, `mbm_head.randomize_temperature`, `mbm_head.tokens_allocation`

### Data Pipeline (`data/webdataset_reader.py`)
- WebDataset format for training data (ImageNet `.tar` shards)
- Supports pretokenized NPZ format for faster generator training

### Training Utilities (`utils/train_utils.py`)
- Central module: model creation, optimizer/scheduler setup, dataloader creation, training loops
- Uses HuggingFace Accelerate for distributed training
- Supports auto-resume, EMA, gradient accumulation, mixed precision (bf16)

### Model Zoo
Pretrained weights downloaded to `assets/` via `huggingface_hub` from `FAR-Amazon/BAR-collections`. Variants: BAR-B/4, BAR-B/2, BAR-B, BAR-L, BAR-L-res512.
