#!/bin/bash

# Distributed training for BAR model using accelerate
# Usage: ./scripts/train_bar_caption.sh

source .env
uv run scripts/train_bar.py \
    config=configs/generator/bar_l_res512_caption.yaml \
    dataset.params.train_shards_path_or_url="$TRAIN_SHARDS_DIR/train-00000.tar" \
    dataset.params.eval_shards_path_or_url="$EVAL_SHARDS_DIR/test-00000.tar" \
    training.gradient_accumulation_steps=1 \
    training.per_gpu_batch_size=1
