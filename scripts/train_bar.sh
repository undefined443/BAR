#!/bin/bash

# Distributed training for BAR model using accelerate
# Usage: ./scripts/train_bar.sh

source .env
uv run accelerate launch \
    --num_machines=1 \
    --num_processes=8 \
    --machine_rank=0 \
    --main_process_ip=10.240.16.7 \
    --main_process_port=9999 \
    scripts/train_bar.py \
    config=configs/generator/bar_b_patch4.yaml \
    dataset.params.train_shards_path_or_url="$TRAIN_SHARDS_DIR/train-{00000..00113}.tar" \
    dataset.params.eval_shards_path_or_url="$EVAL_SHARDS_DIR/test-{00000..00004}.tar" \
    training.gradient_accumulation_steps=1 \
    training.per_gpu_batch_size=$((2048 / 8))
