#!/bin/bash

# Benchmark DDIM sampling quality across different num_steps values.
# Usage: ./scripts/bench_diffusion_steps.sh <checkpoint.bin> [extra OmegaConf args...]
#
# Example:
#   ./scripts/bench_diffusion_steps.sh bar_b_100000/unwrapped_model/pytorch_model.bin
#   ./scripts/bench_diffusion_steps.sh bar_b_100000/unwrapped_model/pytorch_model.bin bench.steps=[1,5,10,20]

set -euo pipefail

source .env

CHECKPOINT="${1:?Usage: $0 <checkpoint.bin> [extra args...]}"
shift

uv run accelerate launch \
    --num_machines=1 \
    --num_processes=1 \
    scripts/bench_diffusion_steps.py \
    config=configs/generator/bar_b.yaml \
    dataset.params.eval_shards_path_or_url="$EVAL_SHARDS_DIR/test-{00000..00004}.tar" \
    dataset.params.num_workers_per_gpu=12 \
    experiment.generator_checkpoint="$CHECKPOINT" \
    "$@"
