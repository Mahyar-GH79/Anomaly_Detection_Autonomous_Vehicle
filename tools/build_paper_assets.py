"""
Build paper-ready assets for the NeurIPS D&B / ICML-style submission.

Produces:
  paper_assets/figures/   – every figure, renamed and grouped by section
  paper_assets/tables/    – LaTeX tables (booktabs) ready to \input{}

Tables generated:
  01_dataset_stats.tex                – per-class counts + splits
  02_binary_results.tex               – VLM + CLIP + SigLIP + Autoencoder
  03_multiclass_results.tex           – VLM + CLIP + SigLIP
  04_description_results.tex          – VLM only (BLEU, METEOR, ROUGE, BERTScore, latency)
  05_calibration.tex                  – ECE, MCE, overconfidence, mean conf, mean acc
  06_consistency_multiclass.tex       – majority-vote acc, mean consistency, entropy, ECE
  07_per_class_f1_full.tex            – per-class F1 across all models (heatmap → table form)
"""

import json
import shutil
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
EVAL_OUT   = ROOT / "vlm_eval_outputs"
ASSETS     = ROOT / "paper_assets"
FIG_DIR    = ASSETS / "figures"
TAB_DIR    = ASSETS / "tables"
DS_JSON    = ROOT / "Data" / "dataset.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
VLMs = [
    "internvl3_1b", "internvl3_2b", "internvl3_8b",
    "qwen2vl_2b",   "qwen25vl_3b",  "qwen25vl_7b",
    "llava_onevision_7b", "llava_13b", "llama32_11b",
]
DISPLAY_VLM = {
    "internvl3_1b":       "InternVL3-1B",
    "internvl3_2b":       "InternVL3-2B",
    "internvl3_8b":       "InternVL3-8B",
    "qwen2vl_2b":         "Qwen2-VL-2B",
    "qwen25vl_3b":        "Qwen2.5-VL-3B",
    "qwen25vl_7b":        "Qwen2.5-VL-7B",
    "llava_onevision_7b": "LLaVA-OV-7B",
    "llava_13b":          "LLaVA-1.6-13B",
    "llama32_11b":        "LLaMA-3.2-11B",
}
CLIP_VARIANTS = [
    ("vit_b32", "CLIP ViT-B/32",  "openai/clip-vit-base-patch32"),
    ("vit_b16", "CLIP ViT-B/16",  "openai/clip-vit-base-patch16"),
    ("vit_l14", "CLIP ViT-L/14",  "openai/clip-vit-large-patch14"),
]
SIGLIP_VARIANTS = [
    ("base_patch16_224",  "SigLIP Base/16",   "google/siglip-base-patch16-224"),
    ("large_patch16_256", "SigLIP Large/16",  "google/siglip-large-patch16-256"),
    ("so400m_patch14_384","SigLIP SO400m/14", "google/siglip-so400m-patch14-384"),
]
ANOMALY_CLASSES = [
    "animal_on_road", "extreme_weather", "road_surface_hazard",
    "fallen_debris_or_vegetation", "strange_object_on_road",
    "vehicle_incident", "infrastructure_failure", "human_presence_anomaly",
    "adverse_lighting", "oversized_or_unusual_vehicle", "multi_hazard_compound",
]
CLASS_DISPLAY = {
    "animal_on_road":              "Animal",
    "extreme_weather":             "Ext.~Weather",
    "road_surface_hazard":         "Road Hazard",
    "fallen_debris_or_vegetation": "Debris/Veg.",
    "strange_object_on_road":      "Strange Obj.",
    "vehicle_incident":            "Vehicle Inc.",
    "infrastructure_failure":      "Infra. Fail.",
    "human_presence_anomaly":      "Human Pres.",
    "adverse_lighting":            "Adv. Light.",
    "oversized_or_unusual_vehicle":"Oversized Veh.",
    "multi_hazard_compound":       "Multi-Hazard",
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def load_json(p):
    if not Path(p).exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt(x, prec=3):
    if x is None or (isinstance(x, float) and (x != x)):
        return "--"
    if isinstance(x, (int,)):
        return f"{x:,}"
    return f"{float(x):.{prec}f}"


def bold_best(values, fmt_fn=lambda v: fmt(v, 3), higher_better=True):
    """Return list of LaTeX-formatted strings; bold the best value."""
    nums = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    if not nums:
        return [fmt_fn(v) for v in values]
    best = max(nums) if higher_better else min(nums)
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and v != v):
            out.append("--")
        elif abs(float(v) - float(best)) < 1e-9:
            out.append(f"\\textbf{{{fmt_fn(v)}}}")
        else:
            out.append(fmt_fn(v))
    return out


def copy_fig(src, dst_name):
    """Copy both .pdf and .png if available."""
    for ext in (".pdf", ".png"):
        s = Path(str(src) + ext) if not str(src).endswith(ext) else Path(src)
        if s.exists():
            shutil.copy2(s, FIG_DIR / f"{dst_name}{s.suffix}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. ORGANISE FIGURES
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Copying figures into paper_assets/figures/ ──")
FIG_MAP = [
    # (source_basename_no_ext, target_basename)
    (EVAL_OUT / "binary"     / "aggregate" / "metrics_comparison",          "binary_metrics_comparison"),
    (EVAL_OUT / "binary"     / "aggregate" / "roc_curve_all_models",        "binary_roc_all_models"),
    (EVAL_OUT / "binary"     / "aggregate" / "pr_curve_all_models",         "binary_pr_all_models"),
    (EVAL_OUT / "multiclass" / "aggregate" / "metrics_comparison",          "multiclass_metrics_comparison"),
    (EVAL_OUT / "multiclass" / "aggregate" / "per_class_f1_heatmap",        "multiclass_per_class_f1_heatmap"),
    (EVAL_OUT / "description"/ "aggregate" / "metrics_comparison",          "description_metrics_comparison"),
    # Per-class analysis
    (EVAL_OUT / "analysis" / "per_class_f1_heatmap",                        "per_class_f1_heatmap"),
    (EVAL_OUT / "analysis" / "hardest_classes",                             "hardest_classes"),
    (EVAL_OUT / "analysis" / "detection_vs_classification",                 "detection_vs_classification"),
    (EVAL_OUT / "analysis" / "per_class_precision_recall",                  "per_class_precision_recall"),
    # Agreement
    (EVAL_OUT / "analysis" / "agreement" / "model_agreement_matrix",        "agreement_matrix"),
    (EVAL_OUT / "analysis" / "agreement" / "ensemble_upper_bound",          "ensemble_upper_bound"),
    (EVAL_OUT / "analysis" / "agreement" / "universal_hard_easy",           "agreement_distribution"),
    (EVAL_OUT / "analysis" / "agreement" / "agreement_by_class",            "agreement_by_class"),
    (EVAL_OUT / "analysis" / "agreement" / "hard_easy_correctness",         "hard_easy_correctness"),
    # Calibration
    (EVAL_OUT / "analysis" / "calibration" / "reliability_diagrams",        "reliability_diagrams"),
    (EVAL_OUT / "analysis" / "calibration" / "ece_comparison",              "calibration_ece"),
    (EVAL_OUT / "analysis" / "calibration" / "confidence_histogram",        "confidence_distributions"),
    (EVAL_OUT / "analysis" / "calibration" / "confidence_vs_accuracy",      "confidence_vs_accuracy"),
    # Vision-encoder representations
    (EVAL_OUT / "analysis" / "representations" / "umap_by_class",           "vision_umap_by_class"),
    (EVAL_OUT / "analysis" / "representations" / "umap_correct_vs_wrong",   "vision_umap_correct_wrong_multiclass"),
    (EVAL_OUT / "analysis" / "representations" / "umap_binary_correct_vs_wrong", "vision_umap_correct_wrong_binary"),
    (EVAL_OUT / "analysis" / "representations" / "umap_by_model",           "vision_umap_by_model"),
    (EVAL_OUT / "analysis" / "representations" / "cka_matrix",              "cka_matrix"),
    (EVAL_OUT / "analysis" / "representations" / "layer_drift",             "layer_drift_internvl3_8b"),
    (EVAL_OUT / "analysis" / "representations" / "linear_probe_accuracy",   "linear_probe_internvl3_8b"),
    # Layer analysis (all models)
    (EVAL_OUT / "analysis" / "layer_analysis" / "layer_drift_all_models",   "layer_drift_all_models"),
    (EVAL_OUT / "analysis" / "layer_analysis" / "layer_drift_overlay",      "layer_drift_overlay"),
    (EVAL_OUT / "analysis" / "layer_analysis" / "linear_probe_all_models",  "linear_probe_all_models"),
    (EVAL_OUT / "analysis" / "layer_analysis" / "linear_probe_overlay",     "linear_probe_overlay"),
]
for src, dst in FIG_MAP:
    copy_fig(src, dst)

# Per-model LLM layer PCA plots — copy the whole folder
LLM_PCA = ASSETS / "figures" / "llm_layer_pca"
(LLM_PCA / "binary").mkdir(parents=True, exist_ok=True)
(LLM_PCA / "multiclass").mkdir(parents=True, exist_ok=True)
for task in ("binary", "multiclass"):
    src_dir = EVAL_OUT / "analysis" / "llm_layers" / task
    if src_dir.exists():
        for f in src_dir.glob("*.pdf"):
            shutil.copy2(f, LLM_PCA / task / f.name)
        for f in src_dir.glob("*.png"):
            shutil.copy2(f, LLM_PCA / task / f.name)
print(f"Copied {len(list(FIG_DIR.glob('*.pdf')))} top-level PDFs and "
      f"{sum(1 for _ in (LLM_PCA).rglob('*.pdf'))} layer-PCA PDFs")


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATASET STATISTICS TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Generating Table 1: Dataset Statistics ──")
ds = load_json(DS_JSON)
samples = ds.get("samples", ds)

n_total       = sum(1 for r in samples.values() if isinstance(r, dict))
n_anomalous   = sum(1 for r in samples.values()
                    if isinstance(r, dict) and r.get("anomaly_present"))
n_normal      = n_total - n_anomalous
class_counts  = {c: 0 for c in ANOMALY_CLASSES}
for r in samples.values():
    if isinstance(r, dict) and r.get("anomaly_present"):
        ac = r.get("anomaly_class")
        if ac in class_counts:
            class_counts[ac] += 1

severity_counts = {}
difficulty_counts = {}
for r in samples.values():
    if isinstance(r, dict) and r.get("anomaly_present"):
        sev = r.get("severity", "unknown")
        dif = r.get("benchmark_difficulty", "unknown")
        severity_counts[sev]   = severity_counts.get(sev, 0) + 1
        difficulty_counts[dif] = difficulty_counts.get(dif, 0) + 1

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\caption{Dataset statistics. The benchmark contains "
           f"{n_total:,} dashcam images comprising {n_normal:,} normal and "
           f"{n_anomalous:,} anomalous scenes. Anomalous images are stratified "
           "across 11 distinct hazard categories.}")
tex.append(r"\label{tab:dataset_stats}")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\begin{tabular}{lr}")
tex.append(r"\toprule")
tex.append(r"\textbf{Statistic} & \textbf{Count} \\")
tex.append(r"\midrule")
tex.append(rf"Total images & {n_total:,} \\")
tex.append(rf"\quad Normal scenes        & {n_normal:,} \\")
tex.append(rf"\quad Anomalous scenes     & {n_anomalous:,} \\")
tex.append(r"\midrule")
tex.append(r"\textbf{Anomaly class breakdown} & \\")
for c in ANOMALY_CLASSES:
    tex.append(rf"\quad {CLASS_DISPLAY[c]} & {class_counts[c]:,} \\")
if severity_counts:
    tex.append(r"\midrule")
    tex.append(r"\textbf{Severity (anomalous only)} & \\")
    for s in ["minor", "moderate", "severe", "critical"]:
        if s in severity_counts:
            tex.append(rf"\quad {s.capitalize()} & {severity_counts[s]:,} \\")
if difficulty_counts:
    tex.append(r"\midrule")
    tex.append(r"\textbf{Benchmark difficulty} & \\")
    for d in ["easy", "medium", "hard"]:
        if d in difficulty_counts:
            tex.append(rf"\quad {d.capitalize()} & {difficulty_counts[d]:,} \\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")
(TAB_DIR / "01_dataset_stats.tex").write_text("\n".join(tex) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 3. BINARY RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("── Generating Table 2: Binary Detection Results ──")

bin_rows = []   # (display_name, family, metrics_dict)

# VLMs
for k in VLMs:
    m = load_json(EVAL_OUT / "binary" / k / "metrics.json")
    if m:
        bin_rows.append((DISPLAY_VLM[k], "VLM", m))

# CLIP
for vk, name, _ in CLIP_VARIANTS:
    m = load_json(ROOT / "Baselines" / "Outputs" / "clip_predictions" / vk / "clip_metrics.json")
    if m:
        bin_rows.append((name, "CLIP", m))

# SigLIP
for vk, name, _ in SIGLIP_VARIANTS:
    m = load_json(ROOT / "Baselines" / "Outputs" / "siglip_predictions" / vk / "siglip_metrics.json")
    if m:
        bin_rows.append((name, "SigLIP", m))

# Autoencoder
ae = load_json(ROOT / "Baselines" / "autoencoder_output" / "ae_metrics.json")
if ae:
    ae_metrics = {
        "accuracy":          ae.get("at_optimal_threshold", {}).get("accuracy"),
        "balanced_accuracy": None,
        "f1":                ae.get("at_optimal_threshold", {}).get("f1"),
        "precision":         ae.get("at_optimal_threshold", {}).get("precision"),
        "recall":            ae.get("at_optimal_threshold", {}).get("recall"),
        "mcc":               None,
        "auroc":             ae.get("auroc"),
        "auprc":             ae.get("auprc"),
    }
    bin_rows.append(("Autoencoder", "AE", ae_metrics))

# Build columns and bold-best
cols = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "mcc", "auroc", "auprc"]
col_labels = ["Acc.", "Bal.~Acc.", "F1", "Prec.", "Rec.", "MCC", "AUROC", "AUPRC"]
col_values = {c: [r[2].get(c) for r in bin_rows] for c in cols}
col_formatted = {c: bold_best(col_values[c], higher_better=True) for c in cols}

tex = []
tex.append(r"\begin{table*}[t]")
tex.append(r"\caption{Binary anomaly detection on the full benchmark "
           f"({n_total:,} images, {n_anomalous:,} anomalous). VLMs are evaluated "
           "zero-shot via prompting; CLIP / SigLIP via image-text similarity; "
           "the Autoencoder via reconstruction error. Best per column in \\textbf{bold}.}")
tex.append(r"\label{tab:binary_results}")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\begin{tabular}{ll" + "c" * len(cols) + "}")
tex.append(r"\toprule")
header = ["Model", "Family"] + col_labels
tex.append(" & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\")
tex.append(r"\midrule")

# Group by family
prev_family = None
for i, (name, family, _) in enumerate(bin_rows):
    if prev_family is not None and family != prev_family:
        tex.append(r"\midrule")
    cells = [name, family] + [col_formatted[c][i] for c in cols]
    tex.append(" & ".join(cells) + r" \\")
    prev_family = family
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table*}")
(TAB_DIR / "02_binary_results.tex").write_text("\n".join(tex) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 4. MULTICLASS RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("── Generating Table 3: Multiclass Classification Results ──")

mc_rows = []
for k in VLMs:
    m = load_json(EVAL_OUT / "multiclass" / k / "metrics.json")
    if m:
        mc_rows.append((DISPLAY_VLM[k], "VLM", m))
for vk, name, _ in CLIP_VARIANTS:
    m = load_json(ROOT / "Baselines" / "Outputs" / "clip_predictions_multiclass" / vk / "metrics.json")
    if m:
        mc_rows.append((name, "CLIP", m))
for vk, name, _ in SIGLIP_VARIANTS:
    m = load_json(ROOT / "Baselines" / "Outputs" / "siglip_predictions_multiclass" / vk / "metrics.json")
    if m:
        mc_rows.append((name, "SigLIP", m))

mc_cols   = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
             "macro_precision", "macro_recall", "cohen_kappa"]
mc_labels = ["Acc.", "Bal.~Acc.", "Macro F1", "Wt. F1", "Macro P", "Macro R", "$\\kappa$"]
mc_values = {c: [r[2].get(c) for r in mc_rows] for c in mc_cols}
mc_formatted = {c: bold_best(mc_values[c], higher_better=True) for c in mc_cols}

tex = []
tex.append(r"\begin{table*}[t]")
tex.append(r"\caption{Multiclass anomaly classification on the "
           f"{n_anomalous:,} anomalous images across 11 classes. "
           "VLMs are queried with the class taxonomy; CLIP / SigLIP "
           "compute image-prompt similarity over per-class natural-language "
           "descriptions. Best per column in \\textbf{bold}.}")
tex.append(r"\label{tab:multiclass_results}")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\begin{tabular}{ll" + "c" * len(mc_cols) + "}")
tex.append(r"\toprule")
header = ["Model", "Family"] + mc_labels
tex.append(" & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\")
tex.append(r"\midrule")
prev_family = None
for i, (name, family, _) in enumerate(mc_rows):
    if prev_family is not None and family != prev_family:
        tex.append(r"\midrule")
    cells = [name, family] + [mc_formatted[c][i] for c in mc_cols]
    tex.append(" & ".join(cells) + r" \\")
    prev_family = family
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table*}")
(TAB_DIR / "03_multiclass_results.tex").write_text("\n".join(tex) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 5. DESCRIPTION RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("── Generating Table 4: Anomaly Description Results ──")
desc_rows = []
for k in VLMs:
    m = load_json(EVAL_OUT / "description" / k / "metrics.json")
    if m:
        desc_rows.append((DISPLAY_VLM[k], m))

desc_cols   = ["bleu_1", "bleu_4", "meteor", "rouge_L",
               "bertscore_precision", "bertscore_recall", "bertscore_f1"]
desc_labels = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L",
               "BERT-P", "BERT-R", "BERT-F1"]
desc_values = {c: [r[1].get(c) for r in desc_rows] for c in desc_cols}
desc_formatted = {c: bold_best(desc_values[c], higher_better=True) for c in desc_cols}

# Latency: lower is better, separate bolding
lat_values    = [r[1].get("latency", {}).get("mean_s") for r in desc_rows]
lat_formatted = bold_best(lat_values,
                          fmt_fn=lambda v: f"{float(v):.2f}",
                          higher_better=False)

tex = []
tex.append(r"\begin{table*}[t]")
tex.append(r"\caption{Anomaly description quality on the "
           f"{n_anomalous:,} anomalous images. We compare against GPT-4o "
           "reference descriptions using BLEU, METEOR, ROUGE, and BERTScore. "
           "Mean inference latency on a single RTX 5090 reported in seconds. "
           "Best per column in \\textbf{bold}.}")
tex.append(r"\label{tab:description_results}")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\begin{tabular}{l" + "c" * (len(desc_cols) + 1) + "}")
tex.append(r"\toprule")
header = ["Model"] + desc_labels + ["Latency (s) $\\downarrow$"]
tex.append(" & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\")
tex.append(r"\midrule")
for i, (name, _) in enumerate(desc_rows):
    cells = [name] + [desc_formatted[c][i] for c in desc_cols] + [lat_formatted[i]]
    tex.append(" & ".join(cells) + r" \\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table*}")
(TAB_DIR / "04_description_results.tex").write_text("\n".join(tex) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 6. CALIBRATION TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("── Generating Table 5: Calibration ──")
cal = load_json(EVAL_OUT / "analysis" / "calibration" / "calibration_summary.json")
tex = []
if cal:
    tex.append(r"\begin{table}[t]")
    tex.append(r"\caption{Confidence-calibration metrics for binary anomaly "
               "detection. ECE (Expected Calibration Error) and MCE (Maximum "
               "Calibration Error) measure how well self-reported confidence "
               "tracks accuracy; lower is better. Overconfidence is the "
               "weighted mean of (confidence $-$ accuracy); positive values "
               "indicate the model is more confident than warranted.}")
    tex.append(r"\label{tab:calibration}")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{lccccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Model} & \textbf{ECE} $\downarrow$ & \textbf{MCE} $\downarrow$ "
               r"& \textbf{Overconf.} & \textbf{Mean Conf.} & \textbf{Mean Acc.} \\")
    tex.append(r"\midrule")

    eces  = [cal[k]["ece"]      for k in VLMs if k in cal]
    mces  = [cal[k]["mce"]      for k in VLMs if k in cal]
    eces_f = bold_best(eces, higher_better=False)
    mces_f = bold_best(mces, higher_better=False)
    j = 0
    for k in VLMs:
        if k not in cal:
            continue
        row = cal[k]
        cells = [
            DISPLAY_VLM[k],
            eces_f[j],
            mces_f[j],
            f"{row['overconf']:+.3f}",
            f"{row['mean_conf']:.3f}",
            f"{row['mean_acc']:.3f}",
        ]
        tex.append(" & ".join(cells) + r" \\")
        j += 1
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    (TAB_DIR / "05_calibration.tex").write_text("\n".join(tex) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 7. CONSISTENCY TABLE (multiclass)
# ──────────────────────────────────────────────────────────────────────────────
print("── Generating Table 6: Multiclass Consistency ──")
# vlm_consistency_eval.py uses different model keys for some models
# (e.g. "llava_ov_7b" instead of "llava_onevision_7b")
CONSISTENCY_ALT_KEYS = {"llava_onevision_7b": "llava_ov_7b"}
cons_rows = []
for k in VLMs:
    cm = load_json(EVAL_OUT / "multiclass" / k / "consistency" / "consistency_metrics.json")
    if cm is None and k in CONSISTENCY_ALT_KEYS:
        cm = load_json(EVAL_OUT / "multiclass" / CONSISTENCY_ALT_KEYS[k] / "consistency" / "consistency_metrics.json")
    if cm:
        cons_rows.append((DISPLAY_VLM[k], cm))

if cons_rows:
    cons_cols = ["majority_vote_accuracy", "majority_vote_balanced_acc",
                 "majority_vote_f1", "mean_consistency",
                 "mean_entropy", "ece"]
    cons_labels = ["MV Acc.", "MV Bal.~Acc.", "MV F1",
                   "Mean Consistency", "Mean Entropy $\\downarrow$",
                   "ECE $\\downarrow$"]
    cons_values = {c: [r[1].get(c) for r in cons_rows] for c in cons_cols}
    higher_better = {"majority_vote_accuracy": True,
                     "majority_vote_balanced_acc": True,
                     "majority_vote_f1": True,
                     "mean_consistency": True,
                     "mean_entropy": False,
                     "ece": False}
    cons_formatted = {c: bold_best(cons_values[c], higher_better=higher_better[c])
                      for c in cons_cols}
    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\caption{Multiclass classification consistency under stochastic "
               "decoding ($T=0.7$, 5 runs per image). MV~Acc. = majority-vote "
               "accuracy; Mean Consistency = fraction of runs agreeing with the "
               "majority vote; Entropy = uncertainty across runs.}")
    tex.append(r"\label{tab:consistency_multiclass}")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l" + "c" * len(cons_cols) + "}")
    tex.append(r"\toprule")
    header = ["Model"] + cons_labels
    tex.append(" & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\")
    tex.append(r"\midrule")
    for i, (name, _) in enumerate(cons_rows):
        cells = [name] + [cons_formatted[c][i] for c in cons_cols]
        tex.append(" & ".join(cells) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table*}")
    (TAB_DIR / "06_consistency_multiclass.tex").write_text("\n".join(tex) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 8. PER-CLASS F1 FULL TABLE (multiclass)
# ──────────────────────────────────────────────────────────────────────────────
print("── Generating Table 7: Per-Class F1 (full) ──")
f1_table = {}
for k in VLMs:
    m = load_json(EVAL_OUT / "multiclass" / k / "metrics.json")
    if m:
        pc = m.get("per_class", {})
        f1_table[DISPLAY_VLM[k]] = {c: pc.get(c, {}).get("f1-score") for c in ANOMALY_CLASSES}
for vk, name, _ in CLIP_VARIANTS:
    m = load_json(ROOT / "Baselines" / "Outputs" / "clip_predictions_multiclass" / vk / "metrics.json")
    if m:
        pc = m.get("per_class", {})
        f1_table[name] = {c: pc.get(c, {}).get("f1-score") for c in ANOMALY_CLASSES}
for vk, name, _ in SIGLIP_VARIANTS:
    m = load_json(ROOT / "Baselines" / "Outputs" / "siglip_predictions_multiclass" / vk / "metrics.json")
    if m:
        pc = m.get("per_class", {})
        f1_table[name] = {c: pc.get(c, {}).get("f1-score") for c in ANOMALY_CLASSES}

if f1_table:
    # Bold best per column
    col_best = {}
    for c in ANOMALY_CLASSES:
        vals = [v[c] for v in f1_table.values() if v.get(c) is not None]
        col_best[c] = max(vals) if vals else None

    short_class = [CLASS_DISPLAY[c].split()[0] for c in ANOMALY_CLASSES]

    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\caption{Per-class F1 scores on multiclass anomaly classification. "
               "Best per column in \\textbf{bold}. Class labels abbreviated; "
               "see Table~\\ref{tab:dataset_stats} for full names. "
               "VLMs evaluated zero-shot via prompting; CLIP / SigLIP via "
               "image-prompt cosine similarity.}")
    tex.append(r"\label{tab:per_class_f1}")
    tex.append(r"\centering")
    tex.append(r"\scriptsize")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{tabular}{l" + "c" * len(ANOMALY_CLASSES) + "}")
    tex.append(r"\toprule")
    header = ["Model"] + short_class
    tex.append(" & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\")
    tex.append(r"\midrule")
    for name, scores in f1_table.items():
        cells = [name]
        for c in ANOMALY_CLASSES:
            v = scores.get(c)
            if v is None:
                cells.append("--")
            elif col_best[c] is not None and abs(v - col_best[c]) < 1e-9:
                cells.append(rf"\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        tex.append(" & ".join(cells) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table*}")
    (TAB_DIR / "07_per_class_f1_full.tex").write_text("\n".join(tex) + "\n")

# ──────────────────────────────────────────────────────────────────────────────
print(f"\n  All tables saved to {TAB_DIR}")
print(f"  All figures saved to {FIG_DIR}")
print(f"\n  Tables generated:")
for t in sorted(TAB_DIR.glob("*.tex")):
    print(f"    - {t.name}")
print(f"\n  Figure count:")
print(f"    - top-level figures : {len(list(FIG_DIR.glob('*.pdf')))}")
print(f"    - LLM layer PCAs    : {sum(1 for _ in LLM_PCA.rglob('*.pdf'))}")
