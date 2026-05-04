#!/usr/bin/env bash
# Run binary anomaly detection across all 9 VLMs.
# Usage:  bash scripts/run_binary.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${CUDA_VISIBLE_DEVICES:-0}
echo "[info] running binary task on GPU ${GPU}"

CUDA_VISIBLE_DEVICES=${GPU} python eval/vlm_eval_tasks.py \
    --task     binary \
    --model    all \
    --eval-dir vlm_eval_outputs
