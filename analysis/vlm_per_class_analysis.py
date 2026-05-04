"""
Per-class performance analysis across all models and tasks.
Produces:
  vlm_eval_outputs/analysis/per_class_analysis.json   — full data
  vlm_eval_outputs/analysis/per_class_f1_heatmap.pdf  — F1 heatmap (models × classes)
  vlm_eval_outputs/analysis/binary_metrics_table.pdf  — binary metrics comparison
  vlm_eval_outputs/analysis/description_metrics_table.pdf
  vlm_eval_outputs/analysis/hardest_classes.pdf       — bar chart of mean F1 per class
  vlm_eval_outputs/analysis/model_ranking.pdf         — radar / bar chart
  vlm_eval_outputs/analysis/summary_table.csv         — machine-readable summary
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent / "vlm_eval_outputs"
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "internvl3_1b", "internvl3_2b", "internvl3_8b",
    "qwen2vl_2b", "qwen25vl_3b", "qwen25vl_7b",
    "llava_onevision_7b", "llava_13b", "llama32_11b",
]

DISPLAY = {
    "internvl3_1b":      "InternVL3-1B",
    "internvl3_2b":      "InternVL3-2B",
    "internvl3_8b":      "InternVL3-8B",
    "qwen2vl_2b":        "Qwen2-VL-2B",
    "qwen25vl_3b":       "Qwen2.5-VL-3B",
    "qwen25vl_7b":       "Qwen2.5-VL-7B",
    "llava_onevision_7b":"LLaVA-OV-7B",
    "llava_13b":         "LLaVA-1.6-13B",
    "llama32_11b":       "LLaMA-3.2-11B",
}

CLASS_DISPLAY = {
    "animal_on_road":              "Animal on Road",
    "extreme_weather":             "Extreme Weather",
    "road_surface_hazard":         "Road Surface Hazard",
    "fallen_debris_or_vegetation": "Fallen Debris/Vegetation",
    "strange_object_on_road":      "Strange Object",
    "vehicle_incident":            "Vehicle Incident",
    "infrastructure_failure":      "Infrastructure Failure",
    "human_presence_anomaly":      "Human Presence",
    "adverse_lighting":            "Adverse Lighting",
    "oversized_or_unusual_vehicle":"Oversized Vehicle",
    "multi_hazard_compound":       "Multi-Hazard",
}

CLASSES = list(CLASS_DISPLAY.keys())

# ── Load all data ──────────────────────────────────────────────────────────────
def load_metrics(task, model):
    p = ROOT / task / model / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

binary_data      = {}
multiclass_data  = {}
description_data = {}

for m in MODELS:
    b = load_metrics("binary",      m)
    c = load_metrics("multiclass",  m)
    d = load_metrics("description", m)
    if b: binary_data[m]      = b
    if c: multiclass_data[m]  = c
    if d: description_data[m] = d

print(f"Loaded: {len(binary_data)} binary, {len(multiclass_data)} multiclass, "
      f"{len(description_data)} description models")

# ── 1. Per-class F1 matrix ────────────────────────────────────────────────────
models_with_mc = [m for m in MODELS if m in multiclass_data]
f1_matrix = np.zeros((len(models_with_mc), len(CLASSES)))
recall_matrix    = np.zeros_like(f1_matrix)
precision_matrix = np.zeros_like(f1_matrix)

for i, m in enumerate(models_with_mc):
    pc = multiclass_data[m].get("per_class", {})
    for j, c in enumerate(CLASSES):
        f1_matrix[i, j]        = pc.get(c, {}).get("f1-score",  0.0)
        recall_matrix[i, j]    = pc.get(c, {}).get("recall",    0.0)
        precision_matrix[i, j] = pc.get(c, {}).get("precision", 0.0)

# support (same for all models)
support = {}
first_mc = multiclass_data[models_with_mc[0]]["per_class"]
for c in CLASSES:
    support[c] = int(first_mc.get(c, {}).get("support", 0))

# ── 2. Binary metrics table ───────────────────────────────────────────────────
BIN_METRICS = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "mcc", "auroc", "auprc"]
bin_rows = []
for m in MODELS:
    if m not in binary_data:
        continue
    row = {"model": DISPLAY[m]}
    for k in BIN_METRICS:
        row[k] = binary_data[m].get(k, float("nan"))
    bin_rows.append(row)
bin_df = pd.DataFrame(bin_rows).set_index("model")

# ── 3. Description metrics table ─────────────────────────────────────────────
DESC_METRICS = ["bleu_1", "bleu_4", "meteor", "rouge_1", "rouge_L",
                "bertscore_f1", "bertscore_precision", "bertscore_recall"]
desc_rows = []
for m in MODELS:
    if m not in description_data:
        continue
    row = {"model": DISPLAY[m]}
    for k in DESC_METRICS:
        row[k] = description_data[m].get(k, float("nan"))
    lat = description_data[m].get("latency", {})
    row["latency_mean_s"] = lat.get("mean_s", float("nan"))
    desc_rows.append(row)
desc_df = pd.DataFrame(desc_rows).set_index("model")

# ── 4. Multiclass aggregate table ────────────────────────────────────────────
MC_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
              "macro_precision", "macro_recall", "cohen_kappa"]
mc_rows = []
for m in MODELS:
    if m not in multiclass_data:
        continue
    row = {"model": DISPLAY[m]}
    for k in MC_METRICS:
        row[k] = multiclass_data[m].get(k, float("nan"))
    mc_rows.append(row)
mc_df = pd.DataFrame(mc_rows).set_index("model")

# ── Save summary CSV ──────────────────────────────────────────────────────────
summary_rows = []
for m in MODELS:
    row = {"model": DISPLAY[m], "model_key": m}
    for k in BIN_METRICS:
        row[f"bin_{k}"] = binary_data.get(m, {}).get(k, float("nan"))
    for k in MC_METRICS:
        row[f"mc_{k}"] = multiclass_data.get(m, {}).get(k, float("nan"))
    for k in DESC_METRICS:
        row[f"desc_{k}"] = description_data.get(m, {}).get(k, float("nan"))
    lat = description_data.get(m, {}).get("latency", {})
    row["desc_latency_mean_s"] = lat.get("mean_s", float("nan"))
    # per-class F1
    pc = multiclass_data.get(m, {}).get("per_class", {})
    for c in CLASSES:
        row[f"f1_{c}"] = pc.get(c, {}).get("f1-score", float("nan"))
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("model_key")
summary_df.to_csv(OUT_DIR / "summary_table.csv")
print(f"Saved summary_table.csv ({len(summary_df)} models)")

# ── Save full JSON ────────────────────────────────────────────────────────────
full_json = {
    "models": MODELS,
    "display_names": DISPLAY,
    "classes": CLASSES,
    "class_display": CLASS_DISPLAY,
    "support": support,
    "binary": {m: binary_data[m] for m in binary_data},
    "multiclass": {m: multiclass_data[m] for m in multiclass_data},
    "description": {m: description_data[m] for m in description_data},
    "per_class_f1_matrix": {
        "models": [DISPLAY[m] for m in models_with_mc],
        "classes": CLASSES,
        "values": f1_matrix.tolist(),
    },
}
with open(OUT_DIR / "per_class_analysis.json", "w") as f:
    json.dump(full_json, f, indent=2)
print("Saved per_class_analysis.json")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

PLT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
}
plt.rcParams.update(PLT_STYLE)

model_labels = [DISPLAY[m] for m in models_with_mc]
class_labels = [CLASS_DISPLAY[c] for c in CLASSES]

# ── Figure 1: Per-class F1 heatmap ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(f1_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(class_labels, rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(models_with_mc)))
ax.set_yticklabels(model_labels, fontsize=9)

# Annotate cells
for i in range(len(models_with_mc)):
    for j in range(len(CLASSES)):
        val = f1_matrix[i, j]
        color = "black" if 0.2 < val < 0.8 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color=color)

plt.colorbar(im, ax=ax, label="F1-score", shrink=0.8)
ax.set_title("Per-Class F1-Score — Multiclass Task", fontsize=12, pad=10)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"per_class_f1_heatmap.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved per_class_f1_heatmap")

# ── Figure 2: Hardest classes (mean F1 across models, with std) ──────────────
mean_f1 = f1_matrix.mean(axis=0)
std_f1  = f1_matrix.std(axis=0)
order   = np.argsort(mean_f1)  # ascending = hardest first

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.RdYlGn(mean_f1[order])
bars = ax.barh([class_labels[i] for i in order], mean_f1[order],
               xerr=std_f1[order], color=colors, capsize=4, edgecolor="gray", linewidth=0.5)
# Add support annotation
for k, idx in enumerate(order):
    ax.text(mean_f1[idx] + std_f1[idx] + 0.01, k,
            f"n={support[CLASSES[idx]]}", va="center", fontsize=8, color="gray")
ax.set_xlim(0, 1.0)
ax.set_xlabel("Mean F1-Score (across all models)")
ax.set_title("Anomaly Class Difficulty — Mean F1 ± Std", fontsize=12)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"hardest_classes.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved hardest_classes")

# ── Figure 3: Per-class F1 grouped bar chart (models side by side per class) ─
fig, ax = plt.subplots(figsize=(16, 5))
x    = np.arange(len(CLASSES))
n_m  = len(models_with_mc)
w    = 0.8 / n_m
cmap = plt.cm.tab10

for i, m in enumerate(models_with_mc):
    offset = (i - n_m / 2 + 0.5) * w
    ax.bar(x + offset, f1_matrix[i], width=w,
           label=DISPLAY[m], color=cmap(i / n_m), edgecolor="white", linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(class_labels, rotation=35, ha="right", fontsize=8)
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1-Score")
ax.set_title("Per-Class F1 by Model — Multiclass Task", fontsize=12)
ax.legend(fontsize=7, ncol=3, loc="upper right")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"per_class_f1_grouped.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved per_class_f1_grouped")

# ── Figure 4: Binary metrics comparison ──────────────────────────────────────
KEY_BIN = ["balanced_accuracy", "f1", "mcc", "auroc"]
KEY_BIN_LABELS = ["Balanced Acc.", "F1", "MCC", "AUROC"]
models_bin = [m for m in MODELS if m in binary_data]

fig, axes = plt.subplots(1, len(KEY_BIN), figsize=(14, 4), sharey=True)
for ax, key, label in zip(axes, KEY_BIN, KEY_BIN_LABELS):
    vals = [binary_data[m].get(key, 0) for m in models_bin]
    colors = plt.cm.RdYlGn(np.array(vals) / max(max(vals), 1e-9))
    bars = ax.barh([DISPLAY[m] for m in models_bin], vals,
                   color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(label, fontsize=9)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)

axes[0].set_title("Binary Detection Performance", fontsize=11)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"binary_metrics.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved binary_metrics")

# ── Figure 5: Description metrics comparison ─────────────────────────────────
KEY_DESC = ["bleu_4", "meteor", "rouge_L", "bertscore_f1"]
KEY_DESC_LABELS = ["BLEU-4", "METEOR", "ROUGE-L", "BERTScore F1"]
models_desc = [m for m in MODELS if m in description_data]

fig, axes = plt.subplots(1, len(KEY_DESC), figsize=(14, 4), sharey=True)
for ax, key, label in zip(axes, KEY_DESC, KEY_DESC_LABELS):
    vals = [description_data[m].get(key, 0) for m in models_desc]
    max_v = max(vals) if vals else 1
    colors = plt.cm.RdYlGn(np.array(vals) / max_v)
    bars = ax.barh([DISPLAY[m] for m in models_desc], vals,
                   color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xlabel(label, fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)

axes[0].set_title("Anomaly Description Quality", fontsize=11)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"description_metrics.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved description_metrics")

# ── Figure 6: Model ranking — multiclass macro F1 vs binary balanced acc ──────
models_both = [m for m in MODELS if m in binary_data and m in multiclass_data]
bin_bacc  = [binary_data[m]["balanced_accuracy"] for m in models_both]
mc_mf1    = [multiclass_data[m]["macro_f1"]       for m in models_both]

fig, ax = plt.subplots(figsize=(7, 6))
cmap = plt.cm.tab10
for i, m in enumerate(models_both):
    ax.scatter(bin_bacc[i], mc_mf1[i], s=120, color=cmap(i / len(models_both)),
               zorder=5, label=DISPLAY[m])
    ax.annotate(DISPLAY[m], (bin_bacc[i], mc_mf1[i]),
                textcoords="offset points", xytext=(6, 3), fontsize=7)

ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Random baseline")
ax.set_xlabel("Binary Balanced Accuracy", fontsize=10)
ax.set_ylabel("Multiclass Macro F1", fontsize=10)
ax.set_title("Detection vs. Classification Performance", fontsize=11)
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"detection_vs_classification.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved detection_vs_classification")

# ── Figure 7: Precision–Recall per class heatmaps (side by side) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 5))
for ax, mat, title in zip(axes,
                           [precision_matrix, recall_matrix],
                           ["Precision", "Recall"]):
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(class_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(models_with_mc)))
    ax.set_yticklabels(model_labels, fontsize=9)
    for i in range(len(models_with_mc)):
        for j in range(len(CLASSES)):
            val = mat[i, j]
            color = "black" if 0.2 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label=title, shrink=0.8)
    ax.set_title(f"Per-Class {title} — Multiclass Task", fontsize=11)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"per_class_precision_recall.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved per_class_precision_recall")

print("\nAll outputs saved to:", OUT_DIR)
print("\n── Multiclass aggregate ──")
print(mc_df.to_string())
print("\n── Binary metrics ──")
print(bin_df.to_string())
print("\n── Description metrics ──")
print(desc_df.to_string())
