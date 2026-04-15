#!/bin/bash

# Distributed training for BAR model using accelerate
# Usage: ./scripts/train_bar_caption.sh

uv run accelerate launch \
    --num_machines=1 \
    --num_processes=8 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=9999 \
    --same_network \
    scripts/train_bar.py \
    config=configs/generator/bar_l_res512_caption.yaml \
    training.gradient_accumulation_steps=1 \
    training.per_gpu_batch_size=$((2048 / 8))
