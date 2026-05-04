"""
Analysis 2 – Confidence Calibration
====================================
Measures whether model-reported confidence scores are reliable predictors
of actual correctness. A model is well-calibrated if, when it says 0.8,
it is correct 80% of the time.

Outputs (vlm_eval_outputs/analysis/calibration/):
  reliability_diagrams.pdf/png   – one subplot per model
  ece_comparison.pdf/png         – ECE bar chart across models
  confidence_histogram.pdf/png   – confidence distribution per model
  calibration_summary.json       – ECE, ACE, MCE, overconfidence stats
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

ROOT    = Path(__file__).resolve().parent.parent / "vlm_eval_outputs"
OUT_DIR = ROOT / "analysis" / "calibration"
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

N_BINS = 10

# ── Load GT ───────────────────────────────────────────────────────────────────
with open(Path(__file__).resolve().parent.parent / "Data" / "dataset.json") as f:
    ds = json.load(f)
gt_raw = ds.get("samples", ds)
gt = {fname: bool(rec.get("anomaly_present", False))
      for fname, rec in gt_raw.items() if isinstance(rec, dict)}

# ── Calibration utilities ─────────────────────────────────────────────────────
def compute_calibration(confidences, corrects, n_bins=10):
    """
    Returns per-bin stats and scalar calibration metrics.
    confidences: array of floats in [0,1]
    corrects: array of bools
    """
    bins       = np.linspace(0, 1, n_bins + 1)
    bin_acc    = np.full(n_bins, np.nan)
    bin_conf   = np.full(n_bins, np.nan)
    bin_count  = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (confidences >= lo) & (confidences < hi if b < n_bins - 1 else confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc[b]   = corrects[mask].mean()
        bin_conf[b]  = confidences[mask].mean()
        bin_count[b] = mask.sum()

    n_total = len(confidences)
    weights = bin_count / n_total

    valid = ~np.isnan(bin_acc)
    ece = np.sum(weights[valid] * np.abs(bin_acc[valid] - bin_conf[valid]))
    mce = np.nanmax(np.abs(bin_acc - bin_conf))

    # Overconfidence: mean(confidence - accuracy) weighted
    overconf = np.sum(weights[valid] * (bin_conf[valid] - bin_acc[valid]))

    # Fraction of predictions at confidence extremes (<=0.55 or >=0.95)
    frac_extreme = ((confidences <= 0.55) | (confidences >= 0.95)).mean()
    frac_high    = (confidences >= 0.9).mean()

    return {
        "bin_acc":    bin_acc,
        "bin_conf":   bin_conf,
        "bin_count":  bin_count,
        "ece":        float(ece),
        "mce":        float(mce),
        "overconf":   float(overconf),
        "frac_extreme": float(frac_extreme),
        "frac_high_conf": float(frac_high),
        "mean_conf":  float(confidences.mean()),
        "mean_acc":   float(corrects.mean()),
    }

# ── Load predictions & compute calibration ───────────────────────────────────
calib = {}
for m in MODELS:
    p = ROOT / "binary" / m / "predictions.json"
    if not p.exists():
        continue
    with open(p) as f:
        data = json.load(f)
    items = data.get("predictions", data)

    confs, correct_list = [], []
    for item in items:
        if item.get("status") != "success":
            continue
        fname = item.get("filename") or item["image_id"] + ".png"
        if fname not in gt:
            continue
        conf  = float(item.get("confidence", 0.5))
        pred  = bool(item.get("anomaly_present", False))
        true  = gt[fname]
        confs.append(conf)
        correct_list.append(pred == true)

    confs   = np.array(confs)
    correct = np.array(correct_list)
    calib[m] = compute_calibration(confs, correct, N_BINS)
    calib[m]["confidences"] = confs
    calib[m]["corrects"]    = correct
    print(f"{DISPLAY[m]:20s}  ECE={calib[m]['ece']:.4f}  "
          f"overconf={calib[m]['overconf']:+.4f}  "
          f"mean_conf={calib[m]['mean_conf']:.3f}  "
          f"mean_acc={calib[m]['mean_acc']:.3f}")

loaded_models = [m for m in MODELS if m in calib]

# ── Save summary JSON ─────────────────────────────────────────────────────────
summary = {
    m: {k: v for k, v in calib[m].items()
        if k not in ("confidences", "corrects", "bin_acc", "bin_conf", "bin_count")}
    for m in loaded_models
}
for m in loaded_models:
    summary[m]["display"] = DISPLAY[m]
with open(OUT_DIR / "calibration_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved calibration_summary.json")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

n_models = len(loaded_models)
ncols = 3
nrows = (n_models + ncols - 1) // ncols

# ── Fig 1: Reliability diagrams (3×3 grid) ───────────────────────────────────
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows))
axes = axes.flatten()

bins_centers = np.linspace(0.05, 0.95, N_BINS)

for idx, m in enumerate(loaded_models):
    ax  = axes[idx]
    cal = calib[m]
    ba  = cal["bin_acc"]
    bc  = cal["bin_conf"]
    cnt = cal["bin_count"]

    valid = ~np.isnan(ba)
    bar_x = np.linspace(0, 1, N_BINS + 1)[:-1]
    bar_w = 1.0 / N_BINS

    # Gap (over/under confidence shading)
    for b in range(N_BINS):
        if not valid[b]:
            continue
        lo = bar_x[b]
        acc_h  = ba[b]
        conf_h = bc[b]
        # perfect calibration bar
        ax.bar(lo, conf_h, width=bar_w, color="#B0BEC5", alpha=0.5,
               align="edge", linewidth=0)
        # actual accuracy bar
        color = "#E53935" if conf_h > acc_h else "#4CAF50"
        ax.bar(lo, acc_h, width=bar_w, color=color, alpha=0.85,
               align="edge", linewidth=0)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_title(f"{DISPLAY[m]}\nECE={cal['ece']:.3f}  over={cal['overconf']:+.3f}",
                 fontsize=8)
    # ECE annotation
    ax.text(0.05, 0.88, f"ECE={cal['ece']:.3f}", transform=ax.transAxes,
            fontsize=8, color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1565C0", alpha=0.8))

# Hide unused axes
for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("Reliability Diagrams — Binary Anomaly Detection\n"
             "(Green = underconfident, Red = overconfident)", fontsize=11, y=1.01)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"reliability_diagrams.{ext}")
plt.close(fig)
print("Saved reliability_diagrams")

# ── Fig 2: ECE comparison bar chart ──────────────────────────────────────────
ece_vals  = [calib[m]["ece"]      for m in loaded_models]
over_vals = [calib[m]["overconf"] for m in loaded_models]
labels    = [DISPLAY[m]           for m in loaded_models]
order     = np.argsort(ece_vals)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

ax = axes[0]
colors = plt.cm.RdYlGn_r(np.array([ece_vals[i] for i in order]) / max(ece_vals))
bars = ax.barh([labels[i] for i in order], [ece_vals[i] for i in order],
               color=colors, edgecolor="gray", linewidth=0.5)
for bar, val in zip(bars, [ece_vals[i] for i in order]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)
ax.set_xlabel("Expected Calibration Error (ECE) ↓")
ax.set_title("Calibration Error by Model")

ax = axes[1]
over_ordered = [over_vals[i] for i in order]
colors2 = ["#E53935" if v > 0 else "#4CAF50" for v in over_ordered]
bars2 = ax.barh([labels[i] for i in order], over_ordered,
                color=colors2, edgecolor="gray", linewidth=0.5, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
for bar, val in zip(bars2, over_ordered):
    xpos = val + 0.003 if val >= 0 else val - 0.003
    ha   = "left" if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=8)
ax.set_xlabel("Overconfidence Score\n(positive = overconfident ↑)")
ax.set_title("Over- vs. Underconfidence")

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"ece_comparison.{ext}")
plt.close(fig)
print("Saved ece_comparison")

# ── Fig 3: Confidence score distributions ────────────────────────────────────
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.0 * nrows), sharey=False)
axes = axes.flatten()
conf_bins = np.linspace(0, 1, 21)

for idx, m in enumerate(loaded_models):
    ax   = axes[idx]
    conf = calib[m]["confidences"]
    corr = calib[m]["corrects"]

    ax.hist(conf[corr],  bins=conf_bins, color="#4CAF50", alpha=0.7,
            label="Correct", density=True)
    ax.hist(conf[~corr], bins=conf_bins, color="#E53935", alpha=0.7,
            label="Wrong",   density=True)
    ax.set_xlabel("Confidence", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(DISPLAY[m], fontsize=8)
    ax.legend(fontsize=7)

for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("Confidence Distributions — Correct vs. Wrong Predictions", fontsize=11, y=1.01)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"confidence_histogram.{ext}")
plt.close(fig)
print("Saved confidence_histogram")

# ── Fig 4: Scatter — mean confidence vs accuracy (per model) ─────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cmap = plt.cm.tab10

# Compute manual offsets to avoid LLaVA-OV-7B / LLaVA-1.6-13B overlap
LABEL_OFFSETS = {
    "internvl3_1b":      (8,  -3),
    "internvl3_2b":      (8,   3),
    "internvl3_8b":      (8,   3),
    "qwen2vl_2b":        (8,  -3),
    "qwen25vl_3b":       (-95, 3),   # left-side
    "qwen25vl_7b":       (8,   3),
    "llava_onevision_7b":(8,  10),   # push up
    "llava_13b":         (8, -12),   # push down
    "llama32_11b":       (8,   3),
}

for i, m in enumerate(loaded_models):
    mc  = calib[m]["mean_conf"]
    acc = calib[m]["mean_acc"]
    color = cmap(i / max(n_models, 1))
    ax.scatter(mc, acc, s=120, color=color, zorder=5, edgecolor="white", linewidth=1.2)
    dx, dy = LABEL_OFFSETS.get(m, (8, 3))
    ha = "right" if dx < 0 else "left"
    ax.annotate(DISPLAY[m], (mc, acc), textcoords="offset points",
                xytext=(dx, dy), fontsize=8, ha=ha, color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=color, alpha=0.85, linewidth=0.6))

lims = [0.0, 1.05]
ax.plot(lims, lims, "k--", linewidth=1.2, alpha=0.6, label="Perfect calibration")
ax.set_xlim(0.15, 1.05)
ax.set_ylim(0.15, 1.0)
ax.set_xlabel("Mean Confidence (self-reported)")
ax.set_ylabel("Actual Accuracy")
ax.set_title("Mean Confidence vs. Actual Accuracy", fontsize=11)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"confidence_vs_accuracy.{ext}")
plt.close(fig)
print("Saved confidence_vs_accuracy")

print(f"\nAll calibration outputs saved to: {OUT_DIR}")
