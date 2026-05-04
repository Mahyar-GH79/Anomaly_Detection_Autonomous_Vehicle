"""
Aggregate PR / ROC curves for the binary anomaly detection task.
Plots all VLMs + CLIP + SigLIP + Autoencoder on shared axes for direct comparison.

Outputs (vlm_eval_outputs/binary/aggregate/):
  pr_curve_all_models.pdf/png
  roc_curve_all_models.pdf/png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

ROOT    = Path(__file__).resolve().parent.parent / "vlm_eval_outputs"
OUT_DIR = ROOT / "binary" / "aggregate"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DS_JSON = Path(__file__).resolve().parent.parent / "Data" / "dataset.json"

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

CLIP_VARIANTS   = [("vit_b32", "CLIP ViT-B/32"),
                   ("vit_b16", "CLIP ViT-B/16"),
                   ("vit_l14", "CLIP ViT-L/14")]
SIGLIP_VARIANTS = [("base_patch16_224",   "SigLIP Base/16-224"),
                   ("large_patch16_256",  "SigLIP Large/16-256"),
                   ("so400m_patch14_384", "SigLIP SO400m/14-384")]

CLIP_BASE   = Path(__file__).resolve().parent.parent / "Baselines" / "Outputs" / "clip_predictions"
SIGLIP_BASE = Path(__file__).resolve().parent.parent / "Baselines" / "Outputs" / "siglip_predictions"
AE_DIR      = Path(__file__).resolve().parent.parent / "Baselines" / "autoencoder_output"

# ── Load GT ───────────────────────────────────────────────────────────────────
with open(DS_JSON) as f:
    ds = json.load(f)
gt_raw = ds.get("samples", ds)
gt = {fname: bool(rec.get("anomaly_present", False))
      for fname, rec in gt_raw.items() if isinstance(rec, dict)}


def load_vlm(model_key):
    p = ROOT / "binary" / model_key / "predictions.json"
    if not p.exists():
        return None, None
    with open(p) as f:
        data = json.load(f)
    items = data.get("predictions", data)
    y_true, y_score = [], []
    for it in items:
        if it.get("status") != "success":
            continue
        fname = it.get("filename") or it["image_id"] + ".png"
        if fname not in gt:
            continue
        # Use confidence as the anomaly score: high conf + pred==anomaly → high anomaly score;
        # high conf + pred==normal → low anomaly score (1-conf)
        conf = float(it.get("confidence", 0.5))
        pred = bool(it.get("anomaly_present", False))
        score = conf if pred else (1.0 - conf)
        y_true.append(int(gt[fname]))
        y_score.append(score)
    return np.array(y_true), np.array(y_score)


def load_clip_siglip(base, variant_key, family):
    p = base / variant_key / f"{family}_predictions.json"
    if not p.exists():
        return None, None
    with open(p) as f:
        data = json.load(f)
    results = data.get("results", data)
    y_true, y_score = [], []
    for r in results:
        y_true.append(int(r["ground_truth_anomaly"]))
        y_score.append(float(r["score_anomalous"]))
    return np.array(y_true), np.array(y_score)


def load_autoencoder():
    csv_path = AE_DIR / "ae_eval_scores.csv"
    json_path = AE_DIR / "ae_eval_scores.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            y_true  = np.array([int(r.get("ground_truth", r.get("gt", 0))) for r in data])
            y_score = np.array([float(r.get("score", r.get("mse", 0))) for r in data])
            return y_true, y_score
    if csv_path.exists():
        import csv
        y_true, y_score = [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column names
                gt_key  = next((k for k in row if k.lower() in ("ground_truth", "gt",
                                "label", "ground_truth_anomaly")), None)
                sc_key  = next((k for k in row if k.lower() in ("score", "mse",
                                "anomaly_score", "reconstruction_error")), None)
                if gt_key and sc_key:
                    y_true.append(int(float(row[gt_key])))
                    y_score.append(float(row[sc_key]))
        if y_true:
            return np.array(y_true), np.array(y_score)
    return None, None


# ── Collect all model curves ─────────────────────────────────────────────────
curves = []
for k in VLMs:
    y_t, y_s = load_vlm(k)
    if y_t is not None:
        curves.append({"name": DISPLAY_VLM[k], "family": "vlm", "y_true": y_t, "y_score": y_s})

for k, name in CLIP_VARIANTS:
    y_t, y_s = load_clip_siglip(CLIP_BASE, k, "clip")
    if y_t is not None:
        curves.append({"name": name, "family": "clip", "y_true": y_t, "y_score": y_s})

for k, name in SIGLIP_VARIANTS:
    y_t, y_s = load_clip_siglip(SIGLIP_BASE, k, "siglip")
    if y_t is not None:
        curves.append({"name": name, "family": "siglip", "y_true": y_t, "y_score": y_s})

ae_t, ae_s = load_autoencoder()
if ae_t is not None:
    curves.append({"name": "Autoencoder", "family": "ae", "y_true": ae_t, "y_score": ae_s})

print(f"Loaded {len(curves)} curves")

# Compute metrics + curves
for c in curves:
    fpr, tpr, _ = roc_curve(c["y_true"], c["y_score"])
    p, r, _     = precision_recall_curve(c["y_true"], c["y_score"])
    c["fpr"]    = fpr
    c["tpr"]    = tpr
    c["recall"] = r
    c["precision"] = p
    c["auroc"]  = roc_auc_score(c["y_true"], c["y_score"])
    c["auprc"]  = average_precision_score(c["y_true"], c["y_score"])

# Sort by AUROC for stable legend ordering
curves.sort(key=lambda c: -c["auroc"])

# ── Colour scheme ─────────────────────────────────────────────────────────────
FAMILY_COLORMAPS = {
    "vlm":    plt.cm.tab10,
    "clip":   plt.cm.Greens,
    "siglip": plt.cm.Purples,
    "ae":     plt.cm.Reds,
}

family_counts = {}
family_idx    = {}
for c in curves:
    family_counts[c["family"]] = family_counts.get(c["family"], 0) + 1
    family_idx[c["family"]]    = 0

for c in curves:
    cmap = FAMILY_COLORMAPS[c["family"]]
    if c["family"] == "vlm":
        c["color"] = cmap(family_idx["vlm"] / max(family_counts["vlm"] - 1, 1))
    else:
        # Skip very pale shades by using 0.45-0.95 range
        n = family_counts[c["family"]]
        t = 0.45 + (family_idx[c["family"]] / max(n - 1, 1)) * 0.5
        c["color"] = cmap(t)
    family_idx[c["family"]] += 1

# ── Fig: ROC curve ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
for c in curves:
    ls = "-" if c["family"] == "vlm" else ("--" if c["family"] == "clip" else
                                            (":" if c["family"] == "siglip" else "-."))
    ax.plot(c["fpr"], c["tpr"], color=c["color"], linewidth=1.6, linestyle=ls,
            label=f"{c['name']} (AUC={c['auroc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Binary Anomaly Detection — ROC Curves (All Models)")
ax.legend(fontsize=7, loc="lower right", ncol=1, framealpha=0.9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"roc_curve_all_models.{ext}")
plt.close(fig)
print("Saved roc_curve_all_models")

# ── Fig: PR curve ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
for c in curves:
    ls = "-" if c["family"] == "vlm" else ("--" if c["family"] == "clip" else
                                            (":" if c["family"] == "siglip" else "-."))
    ax.plot(c["recall"], c["precision"], color=c["color"], linewidth=1.6, linestyle=ls,
            label=f"{c['name']} (AP={c['auprc']:.3f})")

# Random baseline = positive class prevalence
prev = curves[0]["y_true"].mean() if curves else 0.333
ax.axhline(prev, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
           label=f"Random ({prev:.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Binary Anomaly Detection — Precision-Recall Curves (All Models)")
ax.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"pr_curve_all_models.{ext}")
plt.close(fig)
print("Saved pr_curve_all_models")

# ── Save aggregate JSON ───────────────────────────────────────────────────────
agg = [{
    "name":   c["name"],
    "family": c["family"],
    "auroc":  round(float(c["auroc"]), 4),
    "auprc":  round(float(c["auprc"]), 4),
} for c in curves]
with open(OUT_DIR / "all_models_pr_roc.json", "w") as f:
    json.dump(agg, f, indent=2)
print("Saved all_models_pr_roc.json")
