#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${CUDA_VISIBLE_DEVICES:-0}
CUDA_VISIBLE_DEVICES=${GPU} python eval/vlm_eval_tasks.py \
    --task     description \
    --model    all \
    --eval-dir vlm_eval_outputs
