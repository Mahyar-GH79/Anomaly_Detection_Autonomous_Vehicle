#!/usr/bin/env bash
# Regenerate every figure and table in the paper from the saved outputs.
# Run AFTER all eval/consistency runs have completed.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/9] per-class summary"
python analysis/vlm_per_class_analysis.py

echo "[2/9] cross-model agreement + ensemble ceiling"
python analysis/analysis_01_agreement.py

echo "[3/9] confidence calibration"
python analysis/analysis_02_calibration.py

echo "[4/9] vision-encoder representations (UMAP / CKA / probes)"
python analysis/analysis_03_representations.py

echo "[5/9] LLM-layer hidden-state PCA   (NEEDS GPU; ~3-4 hrs)"
python analysis/analysis_04_llm_layers.py

echo "[6/9] layer drift + linear probe across all models"
python analysis/analysis_05_layer_analysis.py

echo "[7/9] aggregate binary PR / ROC curves (16 models)"
python analysis/aggregate_binary_curves.py

echo "[8/9] vision-encoder PCA coloured by GT label"
python analysis/make_vision_umap_by_gt_label.py

echo "[9/9] paper assets (LaTeX tables + figure curation)"
python tools/build_paper_assets.py
python tools/build_binary_consistency_assets.py
python tools/build_PAPER_PLOTS.py

echo "Done. Tables in paper_assets/tables/, figures in paper_assets/figures/."
