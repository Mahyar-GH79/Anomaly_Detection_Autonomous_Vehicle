"""
Analysis 5 – Layer Drift & Linear Probe Accuracy for ALL Models
================================================================
Uses the already-saved .npz representation files (no new inference needed).

For each model:
  - Layer drift : mean cosine distance between consecutive LLM vision-encoder
                  layers, separately for correct vs wrong predictions
  - Linear probe: balanced accuracy of a logistic regression trained on each
                  layer's mean-pooled representations to predict anomaly class

Outputs (vlm_eval_outputs/analysis/layer_analysis/):
  layer_drift_all_models.pdf/png        – 3×3 grid, one subplot per model
  layer_drift_overlay.pdf/png           – all models on one axes
  linear_probe_all_models.pdf/png       – 3×3 grid, one subplot per model
  linear_probe_overlay.pdf/png          – all models on one axes
  layer_analysis_summary.json
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

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
OUT_DIR = ROOT / "analysis" / "layer_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "internvl3_1b", "internvl3_2b", "internvl3_8b",
    "qwen2vl_2b",   "qwen25vl_3b",  "qwen25vl_7b",
    "llava_onevision_7b", "llava_13b", "llama32_11b",
]
DISPLAY = {
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

LAYER_SAMPLE = 300   # images for drift analysis
PROBE_SAMPLE = 500   # images for linear probe
CV_FOLDS     = 3

# ── Load GT ───────────────────────────────────────────────────────────────────
with open(Path(__file__).resolve().parent.parent / "Data" / "dataset.json") as f:
    ds = json.load(f)
gt_raw = ds.get("samples", ds)
gt = {}
for fname, rec in gt_raw.items():
    if isinstance(rec, dict):
        gt[fname] = {
            "anomaly_present": rec.get("anomaly_present", False),
            "anomaly_class":   rec.get("anomaly_class", "normal") if rec.get("anomaly_present") else "normal",
        }

# ── Load binary predictions for correct/wrong labels ─────────────────────────
bin_preds = {}
for m in MODELS:
    p = ROOT / "binary" / m / "predictions.json"
    if not p.exists():
        continue
    with open(p) as f:
        data = json.load(f)
    items = data.get("predictions", data)
    bin_preds[m] = {}
    for item in items:
        if item.get("status") != "success":
            continue
        fname = item.get("filename") or item["image_id"] + ".png"
        pred  = bool(item.get("anomaly_present", False))
        true  = gt.get(fname, {}).get("anomaly_present", False)
        bin_preds[m][fname] = (pred == true)

rng = np.random.default_rng(42)

# ── Main analysis loop ────────────────────────────────────────────────────────
drift_results = {}   # model → {drift_correct, drift_wrong, n_layers}
probe_results = {}   # model → {probe_accs, n_layers}

for m in MODELS:
    repr_dir = ROOT / "multiclass" / m / "representations"
    if not repr_dir.exists():
        print(f"[SKIP] {DISPLAY[m]}: no representations found")
        continue

    npz_files = sorted(repr_dir.glob("*.npz"))
    if len(npz_files) < 10:
        print(f"[SKIP] {DISPLAY[m]}: too few files ({len(npz_files)})")
        continue

    print(f"\n── {DISPLAY[m]} ({len(npz_files)} files) ──")

    # Sample image IDs
    all_ids   = [f.stem for f in npz_files]
    sample_n  = min(PROBE_SAMPLE, len(all_ids))
    sampled   = sorted(rng.choice(all_ids, sample_n, replace=False))

    # Load layers for sampled images
    layer_mats = []
    labels     = []
    correct_flags = []

    for img_id in sampled:
        fpath = repr_dir / f"{img_id}.npz"
        try:
            r = np.load(fpath)
            if "layers" not in r:
                continue
            layer_mats.append(r["layers"].astype(np.float32))  # (n_layers, dim)
            fname = img_id + ".png"
            labels.append(gt.get(fname, {}).get("anomaly_class", "normal"))
            correct_flags.append(bin_preds.get(m, {}).get(fname, None))
        except Exception:
            continue

    if len(layer_mats) < 20:
        print(f"  [SKIP] Too few valid samples ({len(layer_mats)})")
        continue

    stack     = np.stack(layer_mats)   # (n_images, n_layers, dim)
    n_images  = stack.shape[0]
    n_layers  = stack.shape[1]
    labels    = np.array(labels)
    correct_arr = np.array([c if c is not None else -1 for c in correct_flags])

    print(f"  Loaded {n_images} images, {n_layers} layers, dim={stack.shape[2]}")

    # ── Layer drift ───────────────────────────────────────────────────────────
    def cosine_dist(A, B):
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return 1.0 - (A * B).sum(axis=1)

    c_mask = correct_arr == 1
    w_mask = correct_arr == 0

    drift_c = []
    drift_w = []
    for l in range(n_layers - 1):
        A = stack[:, l,     :]
        B = stack[:, l + 1, :]
        d = cosine_dist(A, B)
        drift_c.append(d[c_mask].mean() if c_mask.sum() > 0 else np.nan)
        drift_w.append(d[w_mask].mean() if w_mask.sum() > 0 else np.nan)

    drift_results[m] = {
        "drift_correct": drift_c,
        "drift_wrong":   drift_w,
        "n_layers":      n_layers,
        "n_correct":     int(c_mask.sum()),
        "n_wrong":       int(w_mask.sum()),
    }
    print(f"  Drift: {n_layers-1} transitions  "
          f"(correct={c_mask.sum()}, wrong={w_mask.sum()})")

    # ── Linear probe per layer ────────────────────────────────────────────────
    probe_accs = []
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    # Need at least 2 samples per class for CV
    class_counts = {c: (labels == c).sum() for c in unique_classes}
    valid_classes = [c for c, n in class_counts.items() if n >= CV_FOLDS]
    probe_mask = np.array([l in valid_classes for l in labels])

    if probe_mask.sum() < 30:
        print(f"  [SKIP probe] Too few valid samples after class filter")
        probe_results[m] = {"probe_accs": [], "n_layers": n_layers}
        continue

    X_all    = stack[probe_mask]   # (n_valid, n_layers, dim)
    y_all    = labels[probe_mask]

    print(f"  Probe: {X_all.shape[0]} images, {len(valid_classes)} classes")
    for l in range(n_layers):
        X = X_all[:, l, :].astype(np.float32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        clf    = LogisticRegression(max_iter=300, C=1.0, random_state=42,
                                    n_jobs=-1, solver="lbfgs",
                                    multi_class="multinomial")
        scores = cross_val_score(clf, X, y_all, cv=CV_FOLDS,
                                 scoring="balanced_accuracy")
        probe_accs.append(float(scores.mean()))

    probe_results[m] = {"probe_accs": probe_accs, "n_layers": n_layers}
    print(f"  Probe done — final layer acc: {probe_accs[-1]:.3f}")

# ── Save summary JSON ─────────────────────────────────────────────────────────
summary = {}
for m in MODELS:
    dr = drift_results.get(m, {})
    pr = probe_results.get(m, {})
    summary[m] = {
        "display":       DISPLAY[m],
        "n_layers":      dr.get("n_layers") or pr.get("n_layers"),
        "drift_correct": [round(float(v), 6) for v in dr.get("drift_correct", [])],
        "drift_wrong":   [round(float(v), 6) for v in dr.get("drift_wrong", [])],
        "probe_accs":    [round(float(v), 6) for v in pr.get("probe_accs", [])],
    }
with open(OUT_DIR / "layer_analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved layer_analysis_summary.json")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

models_with_drift  = [m for m in MODELS if m in drift_results]
models_with_probe  = [m for m in MODELS if m in probe_results
                      and probe_results[m]["probe_accs"]]
NCOLS, NROWS = 3, 3

# Model colour map (consistent across all figures)
model_cmap   = plt.cm.tab10
model_colors = {m: model_cmap(i / len(MODELS)) for i, m in enumerate(MODELS)}

# ── Fig 1: Layer drift — 3×3 grid ─────────────────────────────────────────────
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(11, 9))
for idx, m in enumerate(models_with_drift):
    ax = axes.flatten()[idx]
    dr = drift_results[m]
    x  = np.arange(1, dr["n_layers"])
    ax.plot(x, dr["drift_correct"], "o-", color="#2E7D32", linewidth=1.8,
            markersize=4, label=f"Correct (n={dr['n_correct']})")
    ax.plot(x, dr["drift_wrong"],   "s--", color="#C62828", linewidth=1.8,
            markersize=4, label=f"Wrong (n={dr['n_wrong']})")
    ax.set_title(DISPLAY[m], fontsize=9)
    ax.set_xlabel("Layer l → l+1", fontsize=8)
    ax.set_ylabel("Cosine distance", fontsize=8)
    ax.legend(fontsize=6)

for i in range(len(models_with_drift), NROWS * NCOLS):
    axes.flatten()[i].set_visible(False)

fig.suptitle("Layer-wise Representation Drift\n"
             "(Cosine distance between consecutive vision-encoder layers;\n"
             " correct vs wrong binary predictions)",
             fontsize=10, y=1.01)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"layer_drift_all_models.{ext}")
plt.close(fig)
print("Saved layer_drift_all_models")

# ── Fig 2: Layer drift — overlay (all models on one axes) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for m in models_with_drift:
    dr    = drift_results[m]
    x     = np.arange(1, dr["n_layers"])
    # Normalise x to [0,1] so models with different depths are comparable
    x_norm = x / (dr["n_layers"] - 1)
    col   = model_colors[m]
    axes[0].plot(x_norm, dr["drift_correct"], "-",  color=col, linewidth=1.8,
                 alpha=0.85, label=DISPLAY[m])
    axes[1].plot(x_norm, dr["drift_wrong"],   "--", color=col, linewidth=1.8,
                 alpha=0.85, label=DISPLAY[m])

for ax, title in zip(axes, ["Correct Predictions", "Wrong Predictions"]):
    ax.set_xlabel("Relative layer depth (0=first, 1=last)")
    ax.set_ylabel("Mean cosine distance (layer drift)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2)

fig.suptitle("Layer-wise Representation Drift — All Models", fontsize=11)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"layer_drift_overlay.{ext}")
plt.close(fig)
print("Saved layer_drift_overlay")

# ── Fig 3: Linear probe — 3×3 grid ────────────────────────────────────────────
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(11, 9))
for idx, m in enumerate(models_with_probe):
    ax  = axes.flatten()[idx]
    pr  = probe_results[m]
    acc = pr["probe_accs"]
    x   = np.arange(len(acc))
    n_classes_approx = len(set(labels)) if "labels" in dir() else 11
    random_baseline  = 1.0 / n_classes_approx

    ax.plot(x, acc, "o-", color=model_colors[m], linewidth=1.8, markersize=4)
    ax.fill_between(x, acc, alpha=0.12, color=model_colors[m])
    ax.axhline(random_baseline, color="gray", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Random ({random_baseline:.2f})")
    ax.set_ylim(0, max(max(acc) * 1.15, random_baseline * 2))
    ax.set_title(DISPLAY[m], fontsize=9)
    ax.set_xlabel("Layer index", fontsize=8)
    ax.set_ylabel("Balanced acc. (3-fold CV)", fontsize=8)
    ax.legend(fontsize=6)

for i in range(len(models_with_probe), NROWS * NCOLS):
    axes.flatten()[i].set_visible(False)

fig.suptitle("Linear Probe: Anomaly Class Separability per Layer\n"
             "(Logistic regression balanced accuracy; higher = more class-discriminative)",
             fontsize=10, y=1.01)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"linear_probe_all_models.{ext}")
plt.close(fig)
print("Saved linear_probe_all_models")

# ── Fig 4: Linear probe — overlay ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
for m in models_with_probe:
    pr  = probe_results[m]
    acc = pr["probe_accs"]
    x_norm = np.linspace(0, 1, len(acc))
    ax.plot(x_norm, acc, "-", color=model_colors[m], linewidth=2,
            alpha=0.85, label=DISPLAY[m])

ax.axhline(1.0 / 11, color="gray", linestyle="--", linewidth=1,
           alpha=0.7, label="Random baseline (1/11)")
ax.set_xlabel("Relative layer depth (0=first, 1=last)")
ax.set_ylabel("Balanced Accuracy (3-fold CV)")
ax.set_title("Linear Probe: Anomaly Class Separability Across Layers — All Models",
             fontsize=10)
ax.legend(fontsize=7, ncol=2, loc="upper left")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"linear_probe_overlay.{ext}")
plt.close(fig)
print("Saved linear_probe_overlay")

print(f"\nAll layer analysis outputs saved to: {OUT_DIR}")
