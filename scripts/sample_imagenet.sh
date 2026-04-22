#!/bin/bash

# Sample captions from a trained BAR model on ImageNet eval set
# Usage: ./scripts/sample_imagenet.sh

source .env
uv run torchrun \
    --standalone \
    --nproc-per-node=gpu \
    scripts/sample_imagenet.py \
    config=configs/generator/bar_b.yaml \
    dataset.params.eval_shards_path_or_url="$EVAL_SHARDS_DIR/test-{00000..00004}.tar" \
    dataset.params.num_workers_per_gpu=12 \
    experiment.output_dir=bar_b \
    experiment.generator_checkpoint=bar_b/checkpoint-latest/unwrapped_model/pytorch_model.bin \
    model.generator.guidance_scale=9.6 \
    model.generator.mbm_head.randomize_temperature=1.4 \
    "model.generator.mbm_head.tokens_allocation=[64,64,64,64]"
