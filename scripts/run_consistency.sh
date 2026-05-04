#!/usr/bin/env bash
# Stochastic-decoding consistency: T=0.7, N=5 runs/image
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-multiclass}     # binary | multiclass
GPU=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=${GPU} python eval/vlm_consistency_eval.py \
    --task         "${TASK}" \
    --model        all \
    --eval-dir     vlm_eval_outputs \
    --dataset-json Data/dataset.json \
    --images-dir   Data/images
