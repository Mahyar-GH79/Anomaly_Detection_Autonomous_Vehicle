# Reproducing the DASHA-15K Results

This document walks through reproducing every number, table, and figure in
the paper from scratch. Total compute budget is approximately
**845 GPU-hours** on a single NVIDIA RTX 5090 (32 GB).

---

## 0. Environment

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

The repository expects to be run from the project root. All scripts use
`Path(__file__).resolve().parent.parent` to locate `Data/` and
`vlm_eval_outputs/`.

## 1. Get the dataset

The 15,000 dashcam images and annotations are being prepared for a
**separate dataset release** (see [`docs/DATASET.md`](DATASET.md)) and are
not included in this repository. Once the dataset is published, place it
under `./Data/` so that the layout looks like:

```
Data/
├── images/          # 15,000 PNG files (data_00001.png … data_15000.png)
└── dataset.json     # ground-truth annotations
```

All scripts in this repository assume that exact layout.

## 2. Run the three evaluation tasks

### 2.1 Binary anomaly detection (15 K images per model)

```bash
# All 9 VLMs at once
python eval/vlm_eval_tasks.py --task binary --model all \
    --eval-dir vlm_eval_outputs

# Or one at a time
python eval/vlm_eval_tasks.py --task binary --model qwen25vl_3b \
    --eval-dir vlm_eval_outputs
```

Runtime: ~96 GPU-hours total across all 9 VLMs.

### 2.2 Multiclass classification (5 K anomalous per model)

```bash
python eval/vlm_eval_tasks.py --task multiclass --model all \
    --eval-dir vlm_eval_outputs
```

Runtime: ~32 GPU-hours.

### 2.3 Free-form description (5 K anomalous per model)

```bash
python eval/vlm_eval_tasks.py --task description --model all \
    --eval-dir vlm_eval_outputs
```

Runtime: ~45 GPU-hours.

### 2.4 Visual-similarity baselines (CLIP & SigLIP)

```bash
# Binary
python eval/clip_zero_shot.py    --output-dir Baselines/Outputs/clip_predictions
python eval/siglip_zero_shot.py  --output-dir Baselines/Outputs/siglip_predictions

# Multiclass
python eval/clip_siglip_multiclass.py --family all
```

### 2.5 Reconstruction autoencoder baseline

```bash
python eval/autoencoder_train_eval.py
```

## 3. Consistency under stochastic decoding (optional but in paper)

```bash
# Multiclass consistency, all models, T=0.7, N=5
python eval/vlm_consistency_eval.py \
    --task multiclass --model all \
    --eval-dir vlm_eval_outputs \
    --dataset-json Data/dataset.json \
    --images-dir Data/images

# Binary consistency
python eval/vlm_consistency_eval.py \
    --task binary --model all \
    --eval-dir vlm_eval_outputs \
    --dataset-json Data/dataset.json \
    --images-dir Data/images
```

Runtime: ~640 GPU-hours combined (5 forward passes × 9 models × 20K images).

## 4. Cross-LLM dataset validation (optional)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python validation/dataset_validation_claude.py --limit 5000
```

Cost: ~$50 of Anthropic API spend with prompt caching.

## 5. Run all the analyses

```bash
# Per-class summary (CPU-only)
python analysis/vlm_per_class_analysis.py

# Cross-model agreement + ensemble ceiling
python analysis/analysis_01_agreement.py

# Calibration (ECE, reliability, scatter)
python analysis/analysis_02_calibration.py

# Vision-encoder representations (UMAP, CKA, in-script linear probes)
python analysis/analysis_03_representations.py

# LLM-layer hidden-state PCA (per-model figures)
python analysis/analysis_04_llm_layers.py

# Layer drift + linear probe across all models
python analysis/analysis_05_layer_analysis.py

# Aggregate binary PR / ROC curves across all 16 models
python analysis/aggregate_binary_curves.py

# Vision-encoder PCA coloured by binary GT label
python analysis/make_vision_umap_by_gt_label.py
```

## 6. Build paper assets

```bash
python tools/build_paper_assets.py             # Generates LaTeX tables
python tools/build_binary_consistency_assets.py # Binary consistency figure + table
python tools/build_PAPER_PLOTS.py              # Curates the final figure set
```

After this you can compile the paper:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 7. Generate the presentation

```bash
python tools/build_presentation.py
# Output: PAPER_PLOTS/presentation/AV_VLM_Benchmark.pptx
```

---

## Hardware

All numbers in the paper were produced on a workstation with **2× NVIDIA
RTX 5090 (32 GB each)**, 256 GB system RAM, Ubuntu 22.04, Python 3.10,
PyTorch 2.5+, CUDA 12.4. Memory pressure is most severe for InternVL3-8B
and LLaVA-1.6-13B; if you see OOM during the multiclass / consistency
runs, set:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

before launching.
