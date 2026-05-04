"""
Curate the *final* set of figures and tables for the NeurIPS D&B submission
into a single self-contained folder: PAPER_PLOTS/

Layout:
  PAPER_PLOTS/
    main/         – figures meant for the main paper body (limited slots)
    appendix/     – supplementary figures (no page limit)
    tables/       – all LaTeX tables ready to \input{}
    README.md     – maps each file to where it goes in the paper

Curation principles:
  - One figure per major result/story, no redundancy
  - Comparison plots (PR/ROC, heatmaps) preferred over per-model plots in main
  - Per-model LLM layer PCAs go to appendix (too many for main)
  - Keep both PDF (for LaTeX) and PNG (for slides / quick viewing)
"""

import shutil
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
SRC_FIG  = ROOT / "paper_assets" / "figures"
SRC_TAB  = ROOT / "paper_assets" / "tables"
DST      = ROOT / "PAPER_PLOTS"
MAIN     = DST / "main"
APPENDIX = DST / "appendix"
TABLES   = DST / "tables"

for d in (MAIN, APPENDIX, TABLES):
    d.mkdir(parents=True, exist_ok=True)

# ── Main-paper figures (numbered in suggested order of appearance) ───────────
# Format: (source_basename, dest_basename, caption_for_README)
MAIN_FIGURES = [
    ("binary_pr_all_models",            "fig01_binary_pr_curves",
     "Sec. Results — Binary detection PR curves across all 15 models (9 VLMs + 3 CLIP + 3 SigLIP + Autoencoder)."),
    ("binary_roc_all_models",           "fig02_binary_roc_curves",
     "Sec. Results — Binary detection ROC curves across all 15 models."),
    ("per_class_f1_heatmap",            "fig03_multiclass_f1_heatmap",
     "Sec. Results — Per-class F1 heatmap on the multiclass anomaly classification task."),
    ("reliability_diagrams",            "fig04_reliability_diagrams",
     "Sec. Calibration — Reliability diagrams for all 9 VLMs (binary task)."),
    ("calibration_ece",                 "fig05_calibration_ece",
     "Sec. Calibration — ECE and overconfidence comparison across models."),
    ("agreement_matrix",                "fig06_agreement_matrix",
     "Sec. Cross-Model Analysis — Pairwise agreement on binary detection (hierarchical-clustering ordered)."),
    ("ensemble_upper_bound",            "fig07_ensemble_upper_bound",
     "Sec. Cross-Model Analysis — Top-k majority-vote ensemble accuracy curve."),
    ("vision_umap_by_class",            "fig08_vision_umap_by_class",
     "Sec. Representations — Vision encoder PCA of anomalous images coloured by class (best model)."),
    ("vision_umap_correct_wrong_binary","fig09_vision_umap_binary_grid",
     "Sec. Representations — Vision encoder PCA: correct vs wrong binary predictions (3x3 grid, all models)."),
    ("layer_drift_overlay",             "fig10_layer_drift_overlay",
     "Sec. Representations — Layer-wise drift across all models (correct vs wrong, normalised depth)."),
    ("linear_probe_overlay",            "fig11_linear_probe_overlay",
     "Sec. Representations — Linear probe accuracy per layer, all models on shared axes."),
    ("cka_matrix",                      "fig12_cka_matrix",
     "Sec. Representations — Linear CKA between models on anomalous-image representations."),
]

# ── Appendix figures ─────────────────────────────────────────────────────────
APPENDIX_FIGURES = [
    ("agreement_distribution",                "appA01_agreement_distribution",
     "App. — Distribution of per-image agreement (# models correct out of 9)."),
    ("agreement_by_class",                    "appA02_agreement_by_class",
     "App. — Mean per-image agreement stratified by anomaly class."),
    ("hard_easy_correctness",                 "appA03_hard_easy_correctness",
     "App. — Per-model correctness on universal-hard vs universal-easy samples."),
    ("hardest_classes",                       "appA04_hardest_classes",
     "App. — Mean F1 per anomaly class with std across models (class difficulty)."),
    ("detection_vs_classification",           "appA05_detection_vs_classification",
     "App. — Binary balanced acc. vs multiclass macro F1 per model."),
    ("per_class_precision_recall",            "appA06_per_class_precision_recall",
     "App. — Per-class precision and recall heatmaps on multiclass."),
    ("confidence_distributions",              "appA07_confidence_distributions",
     "App. — Confidence histograms per model (correct vs wrong predictions)."),
    ("confidence_vs_accuracy",                "appA08_confidence_vs_accuracy",
     "App. — Mean confidence vs actual accuracy scatter."),
    ("vision_umap_correct_wrong_multiclass",  "appA09_vision_umap_multiclass_grid",
     "App. — Vision encoder PCA: correct vs wrong multiclass predictions (3x3 grid)."),
    ("vision_umap_by_model",                  "appA10_vision_umap_by_model",
     "App. — Combined UMAP of representations across models."),
    ("layer_drift_all_models",                "appA11_layer_drift_grid",
     "App. — Layer drift per model (3x3 grid)."),
    ("linear_probe_all_models",               "appA12_linear_probe_grid",
     "App. — Linear probe accuracy per layer per model (3x3 grid)."),
    ("layer_drift_internvl3_8b",              "appA13_layer_drift_internvl3_8b",
     "App. — Layer drift detail for InternVL3-8B."),
    ("linear_probe_internvl3_8b",             "appA14_linear_probe_internvl3_8b",
     "App. — Linear probe detail for InternVL3-8B."),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def copy_fig(src_basename, dst_basename, dst_dir):
    """Copy both PDF and PNG."""
    moved = []
    for ext in (".pdf", ".png"):
        s = SRC_FIG / f"{src_basename}{ext}"
        if s.exists():
            d = dst_dir / f"{dst_basename}{ext}"
            shutil.copy2(s, d)
            moved.append(d.name)
    return moved


# ── Copy main-paper figures ──────────────────────────────────────────────────
print("\n── Copying MAIN figures ──")
main_copied = []
for src, dst, caption in MAIN_FIGURES:
    moved = copy_fig(src, dst, MAIN)
    if moved:
        main_copied.append((dst, caption, moved))
        print(f"  ✓ {dst}")
    else:
        print(f"  ✗ MISSING: {src}")

# ── Copy appendix figures ────────────────────────────────────────────────────
print("\n── Copying APPENDIX figures ──")
app_copied = []
for src, dst, caption in APPENDIX_FIGURES:
    moved = copy_fig(src, dst, APPENDIX)
    if moved:
        app_copied.append((dst, caption, moved))
        print(f"  ✓ {dst}")
    else:
        print(f"  ✗ MISSING: {src}")

# ── Copy LLM layer PCA plots (selective: representative + best per family) ───
LLM_PCA_DIR = APPENDIX / "llm_layer_pca"
LLM_PCA_BIN = LLM_PCA_DIR / "binary"
LLM_PCA_MC  = LLM_PCA_DIR / "multiclass"
LLM_PCA_BIN.mkdir(parents=True, exist_ok=True)
LLM_PCA_MC.mkdir(parents=True, exist_ok=True)

print("\n── Copying LLM layer PCA plots ──")
llm_src = SRC_FIG / "llm_layer_pca"
llm_count = 0
if llm_src.exists():
    for task_subdir, dst_subdir in (("binary", LLM_PCA_BIN), ("multiclass", LLM_PCA_MC)):
        src_dir = llm_src / task_subdir
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.glob("*")):
            if f.suffix in (".pdf", ".png"):
                shutil.copy2(f, dst_subdir / f.name)
                llm_count += 1
print(f"  Copied {llm_count} LLM layer PCA files")

# ── Copy ALL tables ──────────────────────────────────────────────────────────
print("\n── Copying TABLES ──")
table_copied = []
for tex_path in sorted(SRC_TAB.glob("*.tex")):
    dst = TABLES / tex_path.name
    shutil.copy2(tex_path, dst)
    table_copied.append(tex_path.name)
    print(f"  ✓ {tex_path.name}")

# ── Generate README ──────────────────────────────────────────────────────────
print("\n── Writing README ──")
readme = []
readme.append("# PAPER_PLOTS — NeurIPS Datasets & Benchmarks Submission Assets")
readme.append("")
readme.append("All final figures and tables required for the paper.")
readme.append("Each figure exists in both `.pdf` (for LaTeX) and `.png` (for slides / quick view).")
readme.append("")
readme.append("## Folder layout")
readme.append("```")
readme.append("PAPER_PLOTS/")
readme.append("├── main/         # figures for the main paper body (~12 figures)")
readme.append("├── appendix/     # supplementary figures")
readme.append("│   └── llm_layer_pca/")
readme.append("│       ├── binary/      # 9 per-model layer-wise PCA plots")
readme.append("│       └── multiclass/  # 9 per-model layer-wise PCA plots")
readme.append("└── tables/       # LaTeX tables (.tex files, booktabs format)")
readme.append("```")
readme.append("")

# Tables section
readme.append("## Tables (insert via `\\input{tables/<file>}`)")
readme.append("")
readme.append("| File | Purpose |")
readme.append("|---|---|")
table_purposes = {
    "01_dataset_stats.tex":          "Dataset statistics — total images, per-class counts, severity, difficulty.",
    "02_binary_results.tex":         "Main binary detection results — 15 models × 8 metrics. Headline table.",
    "03_multiclass_results.tex":     "Main multiclass classification results — 15 models × 7 metrics.",
    "04_description_results.tex":    "Anomaly description quality — 9 VLMs, BLEU/METEOR/ROUGE/BERTScore + latency.",
    "05_calibration.tex":            "Confidence calibration — ECE, MCE, overconfidence per VLM.",
    "06_consistency_multiclass.tex": "Multiclass consistency under stochastic decoding (5 runs, T=0.7).",
    "07_per_class_f1_full.tex":      "Per-class F1 across all 15 models (compact appendix table).",
}
for f in sorted(table_copied):
    purpose = table_purposes.get(f, "")
    readme.append(f"| `tables/{f}` | {purpose} |")
readme.append("")

# Main figures
readme.append("## Main figures (`main/`)")
readme.append("")
readme.append("| File | Where it goes |")
readme.append("|---|---|")
for dst, caption, moved in main_copied:
    pdf_name = next((m for m in moved if m.endswith(".pdf")), moved[0])
    readme.append(f"| `main/{pdf_name}` | {caption} |")
readme.append("")

# Appendix figures
readme.append("## Appendix figures (`appendix/`)")
readme.append("")
readme.append("| File | Where it goes |")
readme.append("|---|---|")
for dst, caption, moved in app_copied:
    pdf_name = next((m for m in moved if m.endswith(".pdf")), moved[0])
    readme.append(f"| `appendix/{pdf_name}` | {caption} |")
readme.append("")
readme.append(f"`appendix/llm_layer_pca/` contains {llm_count // 2} per-model layer-wise PCA "
              "figures (PDFs + PNGs), one per (model, task) pair — visualised in the paper as "
              "a representative subset, with the full set referenced for reproducibility.")
readme.append("")

# Quick LaTeX usage
readme.append("## Quick LaTeX usage")
readme.append("```latex")
readme.append("% In your preamble:")
readme.append(r"\usepackage{booktabs}")
readme.append(r"\usepackage{graphicx}")
readme.append("")
readme.append("% Inserting a table:")
readme.append(r"\input{tables/02_binary_results.tex}")
readme.append("")
readme.append("% Inserting a figure:")
readme.append(r"\begin{figure}[t]")
readme.append(r"  \centering")
readme.append(r"  \includegraphics[width=\linewidth]{main/fig03_multiclass_f1_heatmap.pdf}")
readme.append(r"  \caption{Per-class F1 heatmap …}")
readme.append(r"  \label{fig:per_class_f1}")
readme.append(r"\end{figure}")
readme.append("```")
readme.append("")
readme.append("## Notes")
readme.append("- Each `tables/*.tex` is a complete `\\begin{table}…\\end{table}` block — no extra wrapping needed.")
readme.append("- Best per-column values are already pre-bolded in tables.")
readme.append("- All figures use `pdf.fonttype=42` (TrueType embedding) for proper rendering.")

(DST / "README.md").write_text("\n".join(readme) + "\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PAPER_PLOTS built at: {DST}")
print(f"{'='*60}")
print(f"  Main figures      : {len(main_copied)}")
print(f"  Appendix figures  : {len(app_copied)}")
print(f"  LLM layer PCAs    : {llm_count // 2} (× 2 formats)")
print(f"  Tables            : {len(table_copied)}")
print(f"  README            : {DST / 'README.md'}")
