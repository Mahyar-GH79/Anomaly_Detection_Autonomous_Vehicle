#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${CUDA_VISIBLE_DEVICES:-0}
echo "[info] running multiclass task on GPU ${GPU}"

CUDA_VISIBLE_DEVICES=${GPU} python eval/vlm_eval_tasks.py \
    --task     multiclass \
    --model    all \
    --eval-dir vlm_eval_outputs
