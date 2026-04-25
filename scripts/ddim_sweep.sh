#!/bin/bash

# Sweep DDIM num_steps and benchmark caption quality.
# The checkpoint is downloaded from a wandb artifact.
#
# Usage: ./scripts/ddim_sweep.sh <wandb_run_id> <global_step> [extra OmegaConf args...]
#
# Example:
#   ./scripts/ddim_sweep.sh x6zyq2o1 50000
#   ./scripts/ddim_sweep.sh x6zyq2o1 50000 bench.steps=[1,5,10,20]

set -euo pipefail

source .env

RUN_ID="${1:?Usage: $0 <wandb_run_id> <global_step> [extra args...]}"
GLOBAL_STEP="${2:?Usage: $0 <wandb_run_id> <global_step> [extra args...]}"
shift 2

uv run accelerate launch \
    --num_machines=1 \
    --num_processes=8 \
    scripts/ddim_sweep.py \
    config=configs/generator/bar_b.yaml \
    dataset.params.eval_shards_path_or_url="$EVAL_SHARDS_DIR/test-{00000..00004}.tar" \
    dataset.params.num_workers_per_gpu=12 \
    experiment.wandb_run_id="$RUN_ID" \
    experiment.global_step="$GLOBAL_STEP" \
    "$@"
