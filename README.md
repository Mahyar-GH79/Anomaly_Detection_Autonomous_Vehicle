# DASHA-15K: Benchmarking Vision-Language Models for Anomaly Detection in Autonomous Driving

Official code release for the NeurIPS 2026 Datasets & Benchmarks submission
*"Benchmarking Vision-Language Models for Anomaly Detection in Autonomous
Driving."*

DASHA-15K is a benchmark of **15,000 dashcam images** (10,000 normal +
5,000 anomalous spanning **11 hazard categories**) for evaluating
vision-language models on three complementary tasks:

1. **Binary anomaly detection** (15,000 images)
2. **Multiclass classification** of the anomaly type (5,000 anomalous, 11 classes)
3. **Free-form anomaly description** (5,000 anomalous)

We evaluate **9 open-source VLMs** (InternVL3-1B/2B/8B, Qwen2-VL-2B,
Qwen2.5-VL-3B/7B, LLaVA-OneVision-7B, LLaVA-1.6-13B, LLaMA-3.2-11B-Vision)
plus **7 baselines** (3 CLIP variants, 3 SigLIP variants, and a
reconstruction autoencoder), with extensive analyses of per-class
difficulty, confidence calibration, cross-model agreement, representation
geometry, and consistency under stochastic decoding.

---

## Repository layout

```
.
├── eval/                   # Evaluation runners
│   ├── vlm_eval_tasks.py            # Main 9-VLM evaluation across the 3 tasks
│   ├── vlm_consistency_eval.py      # Stochastic-decoding consistency runner
│   ├── clip_zero_shot.py            # CLIP binary baseline
│   ├── siglip_zero_shot.py          # SigLIP binary baseline
│   ├── clip_siglip_multiclass.py    # CLIP + SigLIP multiclass baselines
│   └── autoencoder_train_eval.py    # Reconstruction autoencoder baseline
│
├── analysis/               # Analysis pipelines (regenerate every figure in the paper)
│   ├── vlm_per_class_analysis.py        # Per-class deep dive
│   ├── analysis_01_agreement.py         # Cross-model agreement + ensemble ceiling
│   ├── analysis_02_calibration.py       # ECE, reliability diagrams
│   ├── analysis_03_representations.py   # Vision-encoder UMAP / CKA / probes
│   ├── analysis_04_llm_layers.py        # LLM-layer hidden-state PCA (per model)
│   ├── analysis_05_layer_analysis.py    # Layer drift + linear probes (all models)
│   ├── aggregate_binary_curves.py       # PR / ROC for all 16 models
│   └── make_vision_umap_by_gt_label.py  # Vision-encoder PCA coloured by GT
│
├── validation/             # LLM-as-judge dataset validation
│   └── dataset_validation_claude.py
│
├── tools/                  # Paper asset builders
│   ├── build_paper_assets.py            # Generate LaTeX tables (booktabs)
│   ├── build_PAPER_PLOTS.py             # Curate the final paper figures + tables
│   ├── build_binary_consistency_assets.py
│   └── build_presentation.py            # 16-slide PowerPoint summary
│
├── paper/                  # NeurIPS LaTeX sources
│   ├── main.tex
│   ├── checklist.tex
│   ├── references.bib
│   └── neurips_2026.sty
│
├── docs/                   # Documentation
│   ├── DATASET.md          # Dataset description + datasheet
│   ├── PROMPTS.md          # All prompts used by VLMs and the GPT-4o annotator
│   └── REPRODUCE.md        # End-to-end reproduction instructions
│
├── scripts/                # Bash convenience runners
│   ├── run_binary.sh
│   ├── run_multiclass.sh
│   ├── run_description.sh
│   ├── run_consistency.sh
│   └── run_all_analysis.sh
│
├── README.md               # this file
├── LICENSE                 # MIT (code) + CC-BY 4.0 (dataset, see DATASET.md)
├── requirements.txt
└── .gitignore
```

---

## Quick-start

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/<your-handle>/dasha-15k.git
cd dasha-15k
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# 2. Dataset — NOT included in this repository.
#    The dataset is being prepared for separate release; see
#    docs/DATASET.md for the planned format and licence.
#    Once available, place it under ./Data/ following the layout
#    described in docs/DATASET.md.

# 3. Run a single binary-detection evaluation
python eval/vlm_eval_tasks.py \
    --task   binary \
    --model  qwen25vl_3b \
    --eval-dir vlm_eval_outputs

# 4. Run all 9 VLMs on all 3 tasks
bash scripts/run_binary.sh
bash scripts/run_multiclass.sh
bash scripts/run_description.sh

# 5. Generate every figure and table in the paper
bash scripts/run_all_analysis.sh
python tools/build_paper_assets.py
python tools/build_PAPER_PLOTS.py
```

Full step-by-step instructions are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## Benchmark headlines

- **Best binary detector:** Qwen2.5-VL-3B — accuracy 0.92, MCC 0.82, AUPRC 0.82.
  *Beats 4× larger models, including InternVL3-8B (0.78 acc) and LLaMA-3.2-11B (0.82 acc).*
- **Best multiclass classifier:** InternVL3-8B — macro-F1 0.53, accuracy 0.76.
- **Best calibrated:** Qwen2.5-VL-3B — ECE = 0.027.
- **Most overconfident:** InternVL3-1B — mean confidence 0.95 with accuracy 0.33 (ECE = 0.62).
- **Within-family CKA collinearity:** InternVL3-{1B, 2B, 8B} pairwise CKA > 0.93 — scaling alone does not diversify representations.

See the full results in `paper/main.tex` (Sections 6–8) or in the auto-generated tables under `paper_assets/tables/` after running `tools/build_paper_assets.py`.

---
