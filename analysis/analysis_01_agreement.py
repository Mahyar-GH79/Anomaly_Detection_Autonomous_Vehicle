"""
Analysis 1 – Cross-Model Agreement & Universal Hard/Easy Samples
================================================================
Outputs (vlm_eval_outputs/analysis/agreement/):
  model_agreement_matrix.pdf/png    – pairwise agreement heatmap
  ensemble_upper_bound.pdf/png      – majority-vote accuracy vs. k models
  universal_hard_easy.pdf/png       – breakdown by difficulty / class
  agreement_by_class.pdf/png        – per-class model agreement
  agreement_by_difficulty.pdf/png   – agreement stratified by benchmark_difficulty
  agreement_summary.json            – machine-readable results
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

# ── ICML-style rcParams ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

ROOT    = Path(__file__).resolve().parent.parent / "vlm_eval_outputs"
OUT_DIR = ROOT / "analysis" / "agreement"
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
    "fallen_debris_or_vegetation": "Fallen Debris/Veg.",
    "strange_object_on_road":      "Strange Object",
    "vehicle_incident":            "Vehicle Incident",
    "infrastructure_failure":      "Infrastructure Fail.",
    "human_presence_anomaly":      "Human Presence",
    "adverse_lighting":            "Adverse Lighting",
    "oversized_or_unusual_vehicle":"Oversized Vehicle",
    "multi_hazard_compound":       "Multi-Hazard",
}

# ── Load GT ───────────────────────────────────────────────────────────────────
with open(Path(__file__).resolve().parent.parent / "Data" / "dataset.json") as f:
    ds = json.load(f)
gt_raw = ds.get("samples", ds)
gt = {}   # filename → {anomaly_present, anomaly_class, difficulty, severity, ...}
for fname, rec in gt_raw.items():
    if not isinstance(rec, dict):
        continue
    gt[fname] = {
        "anomaly_present": rec.get("anomaly_present", False),
        "anomaly_class":   rec.get("anomaly_class", "normal"),
        "difficulty":      rec.get("benchmark_difficulty", "unknown"),
        "severity":        rec.get("severity", "none"),
        "ego_risk":        rec.get("ego_risk_level", "unknown"),
    }

# ── Load binary predictions ────────────────────────────────────────────────────
# preds[model][filename] = {"pred": bool, "confidence": float, "correct": bool}
preds = {}
for m in MODELS:
    p = ROOT / "binary" / m / "predictions.json"
    if not p.exists():
        print(f"[WARN] Missing: {p}")
        continue
    with open(p) as f:
        data = json.load(f)
    items = data.get("predictions", data)
    preds[m] = {}
    for item in items:
        fname = item.get("filename") or item.get("image_id") + ".png"
        gt_rec = gt.get(fname, {})
        pred   = bool(item.get("anomaly_present", False))
        true   = bool(gt_rec.get("anomaly_present", False))
        preds[m][fname] = {
            "pred":       pred,
            "confidence": float(item.get("confidence", 0.5)),
            "correct":    pred == true,
            "status":     item.get("status", "success"),
        }

loaded_models = list(preds.keys())
print(f"Loaded {len(loaded_models)} models, {len(gt)} GT records")

# ── Build per-image correctness matrix ────────────────────────────────────────
all_fnames = sorted(gt.keys())
n_images   = len(all_fnames)
n_models   = len(loaded_models)

correct_mat = np.zeros((n_images, n_models), dtype=np.float32)  # 1=correct, 0=wrong, nan=missing
for j, m in enumerate(loaded_models):
    for i, fname in enumerate(all_fnames):
        rec = preds[m].get(fname)
        if rec is None or rec["status"] != "success":
            correct_mat[i, j] = np.nan
        else:
            correct_mat[i, j] = 1.0 if rec["correct"] else 0.0

n_correct_per_image = np.nansum(correct_mat, axis=1)   # how many models got it right
n_valid_per_image   = np.sum(~np.isnan(correct_mat), axis=1)

# ── Universal hard / easy / partial ──────────────────────────────────────────
all_correct_mask  = n_correct_per_image == n_valid_per_image   # all models right
all_wrong_mask    = n_correct_per_image == 0                   # all models wrong
partial_mask      = ~all_correct_mask & ~all_wrong_mask

print(f"Universal easy   (all correct): {all_correct_mask.sum():,}")
print(f"Universal hard   (all wrong)  : {all_wrong_mask.sum():,}")
print(f"Partial                       : {partial_mask.sum():,}")

# Fraction per category
total = n_images
easy_frac    = all_correct_mask.sum() / total
hard_frac    = all_wrong_mask.sum()   / total
partial_frac = partial_mask.sum()     / total

# ── Pairwise model agreement ──────────────────────────────────────────────────
agree_mat = np.zeros((n_models, n_models))
for i, mi in enumerate(loaded_models):
    for j, mj in enumerate(loaded_models):
        mask = ~np.isnan(correct_mat[:, i]) & ~np.isnan(correct_mat[:, j])
        agree = (correct_mat[mask, i] == correct_mat[mask, j]).mean()
        agree_mat[i, j] = agree

# ── Ensemble accuracy (top-k majority vote) ───────────────────────────────────
# Sort models by individual balanced accuracy (use existing metrics)
model_bacc = {}
for m in loaded_models:
    mp = ROOT / "binary" / m / "metrics.json"
    if mp.exists():
        with open(mp) as f:
            model_bacc[m] = json.load(f).get("balanced_accuracy", 0.0)

sorted_models = sorted(loaded_models, key=lambda m: -model_bacc.get(m, 0))

ensemble_results = []
for k in range(1, n_models + 1):
    top_k = sorted_models[:k]
    idxs  = [loaded_models.index(m) for m in top_k]
    votes = np.nanmean(correct_mat[:, idxs], axis=1)  # fraction correct in top-k
    # Majority vote prediction: predict anomaly if majority say so
    pred_mat_k = np.array([
        [preds[loaded_models[j]].get(fname, {}).get("pred", False)
         for j in idxs]
        for fname in all_fnames
    ], dtype=float)
    mv_pred = (pred_mat_k.mean(axis=1) >= 0.5)
    gt_arr  = np.array([gt[f]["anomaly_present"] for f in all_fnames])
    tp = ((mv_pred == True)  & (gt_arr == True)).sum()
    tn = ((mv_pred == False) & (gt_arr == False)).sum()
    fp = ((mv_pred == True)  & (gt_arr == False)).sum()
    fn = ((mv_pred == False) & (gt_arr == True)).sum()
    bacc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    acc  = (tp + tn) / len(gt_arr)
    ensemble_results.append({"k": k, "balanced_accuracy": bacc, "accuracy": acc,
                              "models": [DISPLAY[m] for m in top_k]})

# ── Agreement by anomaly class ────────────────────────────────────────────────
class_agree = defaultdict(list)
for i, fname in enumerate(all_fnames):
    if not gt[fname]["anomaly_present"]:
        continue
    ac = gt[fname]["anomaly_class"]
    n_ok = n_correct_per_image[i]
    n_v  = n_valid_per_image[i]
    if n_v > 0:
        class_agree[ac].append(n_ok / n_v)

class_agree_mean = {c: np.mean(v) for c, v in class_agree.items() if v}
class_agree_std  = {c: np.std(v)  for c, v in class_agree.items() if v}

# ── Agreement by benchmark difficulty ────────────────────────────────────────
diff_agree = defaultdict(list)
for i, fname in enumerate(all_fnames):
    diff = gt[fname]["difficulty"]
    n_ok = n_correct_per_image[i]
    n_v  = n_valid_per_image[i]
    if n_v > 0:
        diff_agree[diff].append(n_ok / n_v)

# ── Save JSON summary ─────────────────────────────────────────────────────────
summary = {
    "n_images":          n_images,
    "n_models":          n_models,
    "universal_easy":    int(all_correct_mask.sum()),
    "universal_hard":    int(all_wrong_mask.sum()),
    "partial":           int(partial_mask.sum()),
    "easy_fraction":     round(easy_frac, 4),
    "hard_fraction":     round(hard_frac, 4),
    "partial_fraction":  round(partial_frac, 4),
    "pairwise_agreement": {
        DISPLAY[loaded_models[i]]: {
            DISPLAY[loaded_models[j]]: round(float(agree_mat[i, j]), 4)
            for j in range(n_models)
        }
        for i in range(n_models)
    },
    "ensemble_balanced_accuracy": [
        {"k": r["k"], "balanced_accuracy": round(r["balanced_accuracy"], 4)}
        for r in ensemble_results
    ],
    "agreement_by_class": {
        c: {"mean": round(v, 4), "std": round(class_agree_std[c], 4)}
        for c, v in class_agree_mean.items()
    },
    "agreement_by_difficulty": {
        d: {"mean": round(np.mean(v), 4), "n": len(v)}
        for d, v in diff_agree.items()
    },
}
with open(OUT_DIR / "agreement_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved agreement_summary.json")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

model_labels = [DISPLAY[m] for m in loaded_models]
CMAP_DIV  = "RdYlGn"
CMAP_SEQ  = "YlOrRd"

# ── Fig 1: Pairwise agreement heatmap (hierarchical-clustering ordered) ──────
# Compute hierarchical clustering order without drawing the dendrogram
# (the dendrogram added visual noise; ordering by similarity is enough).
dist = 1 - agree_mat
np.fill_diagonal(dist, 0)
Z    = linkage(dist[np.triu_indices(n_models, 1)], method="average")
dend = dendrogram(Z, no_plot=True)
order = dend["leaves"]
mat_ordered    = agree_mat[np.ix_(order, order)]
labels_ordered = [model_labels[i] for i in order]

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(mat_ordered, cmap=CMAP_DIV, vmin=0.5, vmax=1.0, aspect="equal")
ax.set_xticks(range(n_models))
ax.set_xticklabels(labels_ordered, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(n_models))
ax.set_yticklabels(labels_ordered, fontsize=8)
for i in range(n_models):
    for j in range(n_models):
        v = mat_ordered[i, j]
        c = "white" if v < 0.65 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=c)

cb = plt.colorbar(im, ax=ax, label="Agreement", shrink=0.85, pad=0.02)
ax.set_title("Pairwise Model Agreement on Binary Anomaly Detection\n"
             "(rows/cols ordered by hierarchical clustering)",
             fontsize=10, pad=8)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"model_agreement_matrix.{ext}")
plt.close(fig)
print("Saved model_agreement_matrix")

# ── Fig 2: Ensemble upper bound ───────────────────────────────────────────────
ks    = [r["k"] for r in ensemble_results]
baccs = [r["balanced_accuracy"] for r in ensemble_results]
accs  = [r["accuracy"] for r in ensemble_results]

# Individual model baccs for reference
indiv_baccs = [model_bacc.get(m, 0) for m in sorted_models]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(ks, baccs, "o-", color="#2196F3", linewidth=2, markersize=6, label="Ensemble (top-k)")
ax.plot(ks, accs,  "s--", color="#FF9800", linewidth=1.5, markersize=5, label="Accuracy")
ax.axhline(max(indiv_baccs), color="#E53935", linestyle=":", linewidth=1.5,
           label=f"Best single model ({max(indiv_baccs):.3f})")
ax.set_xlabel("Number of Models (k, ranked by balanced acc.)")
ax.set_ylabel("Score")
ax.set_xticks(ks)
ax.set_xticklabels([DISPLAY[sorted_models[k-1]] for k in ks],
                    rotation=40, ha="right", fontsize=7)
ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower right")
ax.set_title("Ensemble Majority-Vote Performance", fontsize=10)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"ensemble_upper_bound.{ext}")
plt.close(fig)
print("Saved ensemble_upper_bound")

# ── Fig 3: Universal hard/easy breakdown with difficulty stratification ────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: histogram of "number of models correct" per image
ax = axes[0]
bins  = np.arange(-0.5, n_models + 1.5, 1)
counts, _ = np.histogram(n_correct_per_image, bins=bins)
xs = np.arange(0, n_models + 1)
colors_hist = plt.cm.RdYlGn(xs / n_models)
bars = ax.bar(xs, counts, color=colors_hist, edgecolor="gray", linewidth=0.5)
for x, c in zip(xs, counts):
    if c > 0:
        ax.text(x, c + max(counts) * 0.01,
                f"{c:,}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xs)
ax.set_xlabel(f"# Models Correct (out of {n_models})")
ax.set_ylabel("Number of Images")
ax.set_title(f"Distribution of Per-Image Agreement\n"
             f"(n={n_images:,}; {int(all_correct_mask.sum())} all-correct, "
             f"{int(all_wrong_mask.sum())} all-wrong)",
             fontsize=10)
ax.set_xlim(-0.5, n_models + 0.5)

# Right: per-difficulty breakdown
ax = axes[1]
diff_order = ["easy", "medium", "hard", "unknown"]
diff_labels = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "unknown": "Unknown"}
diff_colors = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#F44336", "unknown": "#9E9E9E"}

diffs_present = [d for d in diff_order if d in diff_agree]
means = [np.mean(diff_agree[d]) for d in diffs_present]
stds  = [np.std(diff_agree[d])  for d in diffs_present]
ns    = [len(diff_agree[d])     for d in diffs_present]
cols  = [diff_colors[d] for d in diffs_present]

bars = ax.bar([diff_labels[d] for d in diffs_present], means,
              yerr=stds, capsize=5, color=cols, edgecolor="gray",
              linewidth=0.5, error_kw=dict(elinewidth=1))
for bar, mean, n in zip(bars, means, ns):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 0.015,
            f"n={n:,}", ha="center", va="bottom", fontsize=7, color="gray")
ax.set_ylabel("Fraction of Models Correct")
ax.set_ylim(0, 1)
ax.set_title("Model Agreement by\nBenchmark Difficulty", fontsize=10)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"universal_hard_easy.{ext}")
plt.close(fig)
print("Saved universal_hard_easy")

# ── Fig 4: Agreement by anomaly class ─────────────────────────────────────────
classes_sorted = sorted(class_agree_mean.keys(), key=lambda c: class_agree_mean[c])
means_c = [class_agree_mean[c] for c in classes_sorted]
stds_c  = [class_agree_std[c]  for c in classes_sorted]
ns_c    = [len(class_agree[c]) for c in classes_sorted]
clabels = [CLASS_DISPLAY.get(c, c) for c in classes_sorted]
bar_colors = plt.cm.RdYlGn(np.array(means_c))

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.barh(clabels, means_c, xerr=stds_c, capsize=4,
               color=bar_colors, edgecolor="gray", linewidth=0.5,
               error_kw=dict(elinewidth=1))
for bar, mean, n in zip(bars, means_c, ns_c):
    ax.text(mean + 0.015, bar.get_y() + bar.get_height()/2,
            f"n={n}", va="center", fontsize=7, color="gray")
ax.set_xlim(0, 1.1)
ax.set_xlabel("Mean Fraction of Models Correct")
ax.axvline(1/n_models, color="#E53935", linestyle=":", linewidth=1.2,
           label=f"Random (1/{n_models})")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
           label="Majority threshold")
ax.legend(fontsize=8)
ax.set_title("Model Agreement by Anomaly Class\n(Binary Detection)", fontsize=10)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"agreement_by_class.{ext}")
plt.close(fig)
print("Saved agreement_by_class")

# ── Fig 5: Per-model correctness heatmap on hard vs easy images ───────────────
# Sample 200 hardest and 200 easiest for a visual overview
hard_idx = np.where(all_wrong_mask)[0]
easy_idx  = np.where(all_correct_mask)[0]
rng = np.random.default_rng(42)
hard_sample = rng.choice(hard_idx, min(150, len(hard_idx)), replace=False)
easy_sample = rng.choice(easy_idx,  min(150, len(easy_idx)),  replace=False)
sel_idx = np.concatenate([np.sort(hard_sample), np.sort(easy_sample)])

mat_sel = correct_mat[sel_idx]  # shape: (300, n_models)
n_hard_show = len(hard_sample)
n_easy_show = len(easy_sample)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(mat_sel.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
               interpolation="nearest")
ax.set_yticks(range(n_models))
ax.set_yticklabels(model_labels, fontsize=8)
ax.set_xlabel("Images (sorted: universal hard | universal easy)", fontsize=9)
ax.axvline(n_hard_show - 0.5, color="white", linewidth=2)
ax.text(n_hard_show / 2, -1.2, "Universal Hard", ha="center", fontsize=8,
        color="#E53935", transform=ax.get_xaxis_transform())
ax.text(n_hard_show + n_easy_show / 2, -1.2, "Universal Easy", ha="center",
        fontsize=8, color="#4CAF50", transform=ax.get_xaxis_transform())
ax.set_xticks([])
plt.colorbar(im, ax=ax, label="Correct (1) / Wrong (0)", shrink=0.8)
ax.set_title("Per-Model Correctness on Universal Hard/Easy Samples", fontsize=10)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"hard_easy_correctness.{ext}")
plt.close(fig)
print("Saved hard_easy_correctness")

print(f"\nAll agreement outputs saved to: {OUT_DIR}")
