"""
Analysis 3 – Hidden State Representation Analysis
===================================================
Uses saved .npz representations (layers, final_rep) to analyse:
  1. UMAP projection coloured by class / model / correct vs wrong
  2. Layer-wise representation drift (how much do representations change
     across layers for correct vs incorrect predictions)
  3. CKA (Centered Kernel Alignment) between models – how similar are
     their learned representations?
  4. Linear probe accuracy per layer – which layer has the most
     class-discriminative information?

Needs GPU for UMAP if dataset is large; falls back to CPU.

Outputs (vlm_eval_outputs/analysis/representations/):
  umap_by_class.pdf/png
  umap_by_model.pdf/png
  umap_correct_vs_wrong.pdf/png
  cka_matrix.pdf/png
  layer_drift.pdf/png
  linear_probe_accuracy.pdf/png
  representation_summary.json
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.2,
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
OUT_DIR = ROOT / "analysis" / "representations"
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
    "animal_on_road":              "Animal",
    "extreme_weather":             "Ext. Weather",
    "road_surface_hazard":         "Road Hazard",
    "fallen_debris_or_vegetation": "Debris/Veg.",
    "strange_object_on_road":      "Strange Obj.",
    "vehicle_incident":            "Vehicle Incident",
    "infrastructure_failure":      "Infra. Fail.",
    "human_presence_anomaly":      "Human Presence",
    "adverse_lighting":            "Adv. Lighting",
    "oversized_or_unusual_vehicle":"Oversized Veh.",
    "multi_hazard_compound":       "Multi-Hazard",
    "normal":                      "Normal",
}

# ── Load GT ───────────────────────────────────────────────────────────────────
with open(Path(__file__).resolve().parent.parent / "Data" / "dataset.json") as f:
    ds = json.load(f)
gt_raw = ds.get("samples", ds)
gt = {}
for fname, rec in gt_raw.items():
    if not isinstance(rec, dict):
        continue
    gt[fname] = {
        "anomaly_present": rec.get("anomaly_present", False),
        "anomaly_class":   rec.get("anomaly_class", "normal") if rec.get("anomaly_present") else "normal",
        "difficulty":      rec.get("benchmark_difficulty", "unknown"),
    }

# ── Load multiclass predictions for correct/wrong labels ─────────────────────
# "correct" = model predicted the right anomaly class (multiclass task)
mc_preds = {}
for m in MODELS:
    p = ROOT / "multiclass" / m / "predictions.json"
    if not p.exists():
        continue
    with open(p) as f:
        data = json.load(f)
    items = data.get("predictions", data)
    mc_preds[m] = {}
    for item in items:
        if item.get("status") != "success":
            continue
        fname      = item.get("filename") or item["image_id"] + ".png"
        pred_class = item.get("scene_class", "unknown")
        true_class = gt.get(fname, {}).get("anomaly_class", "normal")
        mc_preds[m][fname] = {
            "correct":    pred_class == true_class,
            "pred_class": pred_class,
        }

# ── Load binary predictions (kept for layer/drift analysis) ──────────────────
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
        bin_preds[m][fname] = {"correct": pred == true, "pred": pred}

# ── Collect available representation files ────────────────────────────────────
# Use multiclass representations (anomalous images only) for class analysis,
# binary representations for correct/wrong analysis.
# We will sample N_SAMPLE images that appear in ALL models.

N_SAMPLE_UMAP = 600   # images used for UMAP (manageable on CPU)
N_SAMPLE_CKA  = 300   # images used for CKA (n² kernel, keep small)

def get_repr_files(task, model):
    d = ROOT / task / model / "representations"
    if not d.exists():
        return {}
    return {f.stem: f for f in d.glob("*.npz")}

# Multiclass representations (anomalous images, ~5000 per model)
mc_repr = {m: get_repr_files("multiclass", m) for m in MODELS}
mc_repr = {m: v for m, v in mc_repr.items() if v}

# Binary representations (all images, ~15000 per model)
bin_repr = {m: get_repr_files("binary", m) for m in MODELS}
bin_repr = {m: v for m, v in bin_repr.items() if v}

print(f"MC repr: {len(mc_repr)} models, example count: "
      f"{len(next(iter(mc_repr.values()), {}))}")
print(f"Bin repr: {len(bin_repr)} models, example count: "
      f"{len(next(iter(bin_repr.values()), {}))}")

# Common image IDs across all models with MC representations
mc_models = list(mc_repr.keys())
if mc_models:
    common_mc = set(mc_repr[mc_models[0]].keys())
    for m in mc_models[1:]:
        common_mc &= set(mc_repr[m].keys())
    print(f"Common MC images across {len(mc_models)} models: {len(common_mc)}")
else:
    common_mc = set()

bin_models = list(bin_repr.keys())
if bin_models:
    common_bin = set(bin_repr[bin_models[0]].keys())
    for m in bin_models[1:]:
        common_bin &= set(bin_repr[m].keys())
    print(f"Common binary images across {len(bin_models)} models: {len(common_bin)}")
else:
    common_bin = set()

# ── Helper: load final_rep for a set of image_ids ─────────────────────────────
def load_final_reps(model, repr_dict, image_ids):
    reps = []
    for img_id in image_ids:
        try:
            r = np.load(repr_dict[img_id])
            reps.append(r["final_rep"].astype(np.float32))
        except Exception:
            reps.append(None)
    return reps

def load_all_layers(model, repr_dict, image_ids):
    """Returns array (n_images, n_layers, hidden_dim)."""
    all_layers = []
    for img_id in image_ids:
        try:
            r = np.load(repr_dict[img_id])
            all_layers.append(r["layers"].astype(np.float32))
        except Exception:
            all_layers.append(None)
    return all_layers

# ── Sample images for UMAP ────────────────────────────────────────────────────
rng = np.random.default_rng(42)

if len(common_mc) >= N_SAMPLE_UMAP:
    umap_ids = sorted(rng.choice(sorted(common_mc), N_SAMPLE_UMAP, replace=False))
else:
    umap_ids = sorted(common_mc)
print(f"Using {len(umap_ids)} images for UMAP")

# ── Helper: fit dimensionality reduction ─────────────────────────────────────
def fit_reducer(matrix):
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                                random_state=42, metric="cosine",
                                low_memory=False, n_jobs=-1)
        return reducer.fit_transform(matrix), "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(matrix), "PCA"

# ── Per-model embeddings (for 3×3 grid figures) ───────────────────────────────
print("Computing per-model embeddings for 3×3 UMAP grids...")
per_model_emb   = {}   # model → (embedding, ids_valid, classes, mc_correct)
umap_model      = None  # will be set to best MC model for standalone plots
umap_ids_valid  = []

for m in MODELS:
    if m not in mc_repr:
        continue
    raws = load_final_reps(m, mc_repr[m], umap_ids)
    valid_idx  = [i for i, r in enumerate(raws) if r is not None]
    ids_valid  = [umap_ids[i] for i in valid_idx]
    matrix     = np.stack([raws[i] for i in valid_idx]).astype(np.float32)

    classes  = [gt.get(img_id + ".png", {}).get("anomaly_class", "normal")
                for img_id in ids_valid]
    correct  = [mc_preds.get(m, {}).get(img_id + ".png", {}).get("correct", None)
                for img_id in ids_valid]

    emb, method = fit_reducer(matrix)
    per_model_emb[m] = {"emb": emb, "ids": ids_valid,
                        "classes": classes, "correct": correct,
                        "method": method}
    print(f"  {DISPLAY[m]:20s}  {matrix.shape}  → {method}")
    if umap_model is None:
        umap_model     = m
        umap_ids_valid = ids_valid

# Best single-model for standalone class plot: pick highest MC macro_f1
best_mc = max(
    (m for m in MODELS if m in per_model_emb),
    key=lambda m: (ROOT / "multiclass" / m / "metrics.json").exists() and
                  json.load(open(ROOT / "multiclass" / m / "metrics.json")).get("macro_f1", 0),
    default=umap_model,
)

# ── Fig 1a: UMAP by class (best model, standalone) ───────────────────────────
if best_mc and best_mc in per_model_emb:
    emb_data   = per_model_emb[best_mc]
    embedding  = emb_data["emb"]
    umap_classes = emb_data["classes"]
    all_classes  = sorted(set(umap_classes))
    n_classes    = len(all_classes)
    class_cmap   = plt.colormaps["tab20"].resampled(max(n_classes, 1))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in all_classes:
        mask = np.array([c == cls for c in umap_classes])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   s=14, alpha=0.85, color=class_cmap(class_to_idx[cls] / n_classes),
                   label=CLASS_DISPLAY.get(cls, cls), linewidths=0)
    ax.set_xlabel(f"{emb_data['method']}-1")
    ax.set_ylabel(f"{emb_data['method']}-2")
    ax.set_title(f"Representation Space by Anomaly Class\n({DISPLAY[best_mc]})", fontsize=10)
    ax.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"umap_by_class.{ext}")
    plt.close(fig)
    print("Saved umap_by_class")

# ── Fig 1b: 3×3 grid — Correct vs Wrong (multiclass) per model ───────────────
# "Correct" = model predicted the right anomaly class.
# Representations from the multiclass task (anomalous images only).
NCOLS, NROWS = 3, 3
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(12, 10))
axes_flat = axes.flatten()
plot_idx  = 0

for m in MODELS:
    if m not in per_model_emb or plot_idx >= NROWS * NCOLS:
        continue
    ax       = axes_flat[plot_idx]
    emb_data = per_model_emb[m]
    emb      = emb_data["emb"]
    correct  = emb_data["correct"]

    c_mask = np.array([c is True  for c in correct])
    w_mask = np.array([c is False for c in correct])
    n_mask = np.array([c is None  for c in correct])

    # Wrong first (background), correct on top
    if w_mask.sum():
        ax.scatter(emb[w_mask, 0], emb[w_mask, 1],
                   s=12, alpha=0.85, color="#D32F2F",
                   label=f"Wrong ({w_mask.sum()})", linewidths=0, rasterized=True)
    if c_mask.sum():
        ax.scatter(emb[c_mask, 0], emb[c_mask, 1],
                   s=12, alpha=0.85, color="#2E7D32",
                   label=f"Correct ({c_mask.sum()})", linewidths=0, rasterized=True)

    pct = c_mask.sum() / max(c_mask.sum() + w_mask.sum(), 1)
    ax.set_title(f"{DISPLAY[m]}\n(acc={pct:.2f})", fontsize=8, pad=3)
    ax.set_xlabel(emb_data["method"] + "-1", fontsize=7)
    ax.set_ylabel(emb_data["method"] + "-2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, markerscale=1.5, loc="lower right",
              handletextpad=0.3, borderpad=0.4)
    plot_idx += 1

# Hide unused subplots
for i in range(plot_idx, NROWS * NCOLS):
    axes_flat[i].set_visible(False)

fig.suptitle("Representation Space: Correct vs. Wrong Multiclass Predictions\n"
             "(Anomalous images only; colour = model's classification accuracy on each sample)",
             fontsize=10, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 1])
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"umap_correct_vs_wrong.{ext}")
plt.close(fig)
print("Saved umap_correct_vs_wrong (3×3 grid)")

# ── Fig 1c: 3×3 grid — Binary correct vs wrong ────────────────────────────────
# Representations from the BINARY task (all 15,000 images: normal + anomalous).
# "Correct" = model correctly predicted whether the image is anomalous or normal.
# Sampled 300 normal + 300 anomalous per model for a balanced view.

N_BIN_SAMPLE = 300   # per class (normal / anomalous)

# Build balanced sample IDs common across all models with binary representations
normal_ids   = sorted(img_id for img_id in common_bin
                      if not gt.get(img_id + ".png", {}).get("anomaly_present", True))
anomalous_ids = sorted(img_id for img_id in common_bin
                       if gt.get(img_id + ".png", {}).get("anomaly_present", False))

bin_sample_normal   = sorted(rng.choice(normal_ids,
                                        min(N_BIN_SAMPLE, len(normal_ids)),
                                        replace=False))
bin_sample_anomalous = sorted(rng.choice(anomalous_ids,
                                          min(N_BIN_SAMPLE, len(anomalous_ids)),
                                          replace=False))
bin_sample_ids = bin_sample_normal + bin_sample_anomalous   # 600 total, balanced
print(f"Binary UMAP sample: {len(bin_sample_normal)} normal + "
      f"{len(bin_sample_anomalous)} anomalous = {len(bin_sample_ids)} images")

NCOLS, NROWS = 3, 3
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(12, 10))
axes_flat  = axes.flatten()
plot_idx   = 0

for m in MODELS:
    if m not in bin_repr or plot_idx >= NROWS * NCOLS:
        continue
    ax = axes_flat[plot_idx]

    raws  = load_final_reps(m, bin_repr[m], bin_sample_ids)
    valid_idx  = [i for i, r in enumerate(raws) if r is not None]
    ids_valid  = [bin_sample_ids[i] for i in valid_idx]
    matrix     = np.stack([raws[i] for i in valid_idx]).astype(np.float32)

    # Labels
    gt_labels = np.array([gt.get(img_id + ".png", {}).get("anomaly_present", False)
                           for img_id in ids_valid])
    pred_correct = np.array([
        bin_preds.get(m, {}).get(img_id + ".png", {}).get("correct", None)
        for img_id in ids_valid
    ])

    # Dimensionality reduction
    emb, method = fit_reducer(matrix)

    c_mask = pred_correct == True
    w_mask = pred_correct == False

    # Wrong first (background), correct on top
    if w_mask.sum():
        ax.scatter(emb[w_mask, 0], emb[w_mask, 1],
                   s=12, alpha=0.85, color="#D32F2F",
                   label=f"Wrong ({w_mask.sum()})", linewidths=0, rasterized=True)
    if c_mask.sum():
        ax.scatter(emb[c_mask, 0], emb[c_mask, 1],
                   s=12, alpha=0.85, color="#2E7D32",
                   label=f"Correct ({c_mask.sum()})", linewidths=0, rasterized=True)

    pct = c_mask.sum() / max(c_mask.sum() + w_mask.sum(), 1)
    ax.set_title(f"{DISPLAY[m]}\n(acc={pct:.2f})", fontsize=8, pad=3)
    ax.set_xlabel(method + "-1", fontsize=7)
    ax.set_ylabel(method + "-2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, markerscale=1.5, loc="lower right",
              handletextpad=0.3, borderpad=0.4)
    plot_idx += 1

for i in range(plot_idx, NROWS * NCOLS):
    axes_flat[i].set_visible(False)

fig.suptitle("Representation Space: Correct vs. Wrong Binary Predictions\n"
             "(300 normal + 300 anomalous images; colour = binary detection accuracy per sample)",
             fontsize=10, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 1])
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"umap_binary_correct_vs_wrong.{ext}")
plt.close(fig)
print("Saved umap_binary_correct_vs_wrong (3×3 grid)")

# ── CKA between models ────────────────────────────────────────────────────────
def linear_cka(X, Y):
    """Compute linear CKA between representation matrices X and Y.
    X, Y: (n, d) float arrays. n must match."""
    def center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K = X @ X.T
    L = Y @ Y.T
    Kc = center(K)
    Lc = center(L)
    hsic = np.sum(Kc * Lc)
    norm = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return float(hsic / (norm + 1e-10))

# Use common MC images for CKA
cka_models = [m for m in MODELS if m in mc_repr]
if len(common_mc) >= N_SAMPLE_CKA and len(cka_models) >= 2:
    cka_ids = sorted(rng.choice(sorted(common_mc), N_SAMPLE_CKA, replace=False))
    print(f"Computing CKA on {len(cka_ids)} images × {len(cka_models)} models...")

    # Load final_reps for all CKA models
    cka_reps = {}
    for m in cka_models:
        raws = load_final_reps(m, mc_repr[m], cka_ids)
        valid = [r for r in raws if r is not None]
        if len(valid) < N_SAMPLE_CKA * 0.8:
            print(f"  [SKIP] {DISPLAY[m]}: too many missing")
            continue
        # Pad missing with zeros (rare)
        filled = [r if r is not None else np.zeros_like(valid[0]) for r in raws]
        mat = np.stack(filled)
        # L2 normalise rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        cka_reps[m] = mat / norms

    cka_models_valid = list(cka_reps.keys())
    n_cka = len(cka_models_valid)
    cka_mat = np.eye(n_cka)
    for i in range(n_cka):
        for j in range(i + 1, n_cka):
            val = linear_cka(cka_reps[cka_models_valid[i]],
                             cka_reps[cka_models_valid[j]])
            cka_mat[i, j] = val
            cka_mat[j, i] = val
            print(f"  CKA({DISPLAY[cka_models_valid[i]]}, "
                  f"{DISPLAY[cka_models_valid[j]]}) = {val:.3f}")

    # ── Fig 2: CKA heatmap ────────────────────────────────────────────────────
    cka_labels = [DISPLAY[m] for m in cka_models_valid]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cka_mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_cka))
    ax.set_xticklabels(cka_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n_cka))
    ax.set_yticklabels(cka_labels, fontsize=8)
    for i in range(n_cka):
        for j in range(n_cka):
            c = "white" if cka_mat[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cka_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=c)
    plt.colorbar(im, ax=ax, label="Linear CKA", shrink=0.85)
    ax.set_title("Representation Similarity between Models\n(Linear CKA on Anomalous Images)",
                 fontsize=10)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"cka_matrix.{ext}")
    plt.close(fig)
    print("Saved cka_matrix")

    # Save CKA values
    cka_summary = {
        cka_labels[i]: {cka_labels[j]: round(float(cka_mat[i, j]), 4)
                        for j in range(n_cka)}
        for i in range(n_cka)
    }
else:
    print("[SKIP] Not enough data for CKA")
    cka_summary = {}
    cka_mat = None

# ── Layer-wise drift analysis ─────────────────────────────────────────────────
# For a single model, compute cosine similarity between consecutive layers
# separately for correct and incorrect predictions.
layer_model = "internvl3_8b" if "internvl3_8b" in mc_repr else (mc_models[0] if mc_models else None)

LAYER_SAMPLE = 200
if layer_model and common_mc:
    layer_ids = sorted(rng.choice(sorted(common_mc),
                                  min(LAYER_SAMPLE, len(common_mc)), replace=False))
    print(f"Loading layer-wise representations for {DISPLAY[layer_model]}...")
    layer_data = load_all_layers(layer_model, mc_repr[layer_model], layer_ids)
    valid_layers = [(img_id, d) for img_id, d in zip(layer_ids, layer_data)
                    if d is not None]

    # correct/wrong label
    correct_labels = []
    for img_id, _ in valid_layers:
        fname   = img_id + ".png"
        correct = bin_preds.get(layer_model, {}).get(fname, {}).get("correct", None)
        correct_labels.append(correct)

    if valid_layers:
        layer_mats = np.stack([d for _, d in valid_layers])  # (n, n_layers, dim)
        n_layers   = layer_mats.shape[1]

        # Cosine sim between consecutive layers, averaged over images
        def cosine_sim(A, B):
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
            return (A * B).sum(axis=1)

        drift_correct = []
        drift_wrong   = []
        c_mask = np.array([c == True  for c in correct_labels])
        w_mask = np.array([c == False for c in correct_labels])

        for l in range(n_layers - 1):
            A = layer_mats[:, l,     :]
            B = layer_mats[:, l + 1, :]
            sim = cosine_sim(A, B)
            if c_mask.sum() > 0:
                drift_correct.append(1 - sim[c_mask].mean())
            else:
                drift_correct.append(np.nan)
            if w_mask.sum() > 0:
                drift_wrong.append(1 - sim[w_mask].mean())
            else:
                drift_wrong.append(np.nan)

        # ── Fig 3: Layer drift ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 3.5))
        layer_x = np.arange(1, n_layers)
        ax.plot(layer_x, drift_correct, "o-", color="#4CAF50", linewidth=2,
                markersize=5, label="Correct predictions")
        ax.plot(layer_x, drift_wrong,   "s--", color="#E53935", linewidth=2,
                markersize=5, label="Wrong predictions")
        ax.set_xlabel("Layer transition (l → l+1)")
        ax.set_ylabel("Mean cosine distance (drift)")
        ax.set_title(f"Layer-wise Representation Drift\n({DISPLAY[layer_model]})", fontsize=10)
        ax.legend()
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"layer_drift.{ext}")
        plt.close(fig)
        print("Saved layer_drift")

# ── Linear probe per layer ────────────────────────────────────────────────────
# Train a logistic regression on each layer's representations to predict
# the true binary label. Accuracy across layers shows which layer is most
# class-discriminative.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

PROBE_SAMPLE = 400
probe_model  = layer_model  # reuse same model

if probe_model and common_mc:
    probe_ids = sorted(rng.choice(sorted(common_mc),
                                  min(PROBE_SAMPLE, len(common_mc)), replace=False))
    probe_data = load_all_layers(probe_model, mc_repr[probe_model], probe_ids)
    valid_probe = [(img_id, d) for img_id, d in zip(probe_ids, probe_data)
                   if d is not None]

    probe_labels = []
    for img_id, _ in valid_probe:
        fname = img_id + ".png"
        ac    = gt.get(fname, {}).get("anomaly_class", "normal")
        probe_labels.append(ac)

    if valid_probe:
        probe_mats = np.stack([d for _, d in valid_probe])  # (n, n_layers, dim)
        n_layers   = probe_mats.shape[1]
        label_arr  = np.array(probe_labels)

        probe_accs = []
        print(f"Running linear probes across {n_layers} layers...")
        for l in range(n_layers):
            X = probe_mats[:, l, :].astype(np.float32)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=200, C=1.0, random_state=42, n_jobs=-1)
            scores = cross_val_score(clf, X, label_arr, cv=3, scoring="balanced_accuracy")
            probe_accs.append(scores.mean())
            print(f"  Layer {l:2d}: balanced_acc = {scores.mean():.3f}")

        # ── Fig 4: Linear probe accuracy per layer ─────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(range(n_layers), probe_accs, "o-", color="#1976D2", linewidth=2,
                markersize=5)
        ax.fill_between(range(n_layers), probe_accs,
                        alpha=0.15, color="#1976D2")
        ax.axhline(1 / len(set(probe_labels)), color="gray", linestyle="--",
                   linewidth=1, label="Random baseline")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Balanced Accuracy (3-fold CV)")
        ax.set_title(f"Linear Probe: Anomaly Class Separability per Layer\n"
                     f"({DISPLAY[probe_model]})", fontsize=10)
        ax.legend()
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"linear_probe_accuracy.{ext}")
        plt.close(fig)
        print("Saved linear_probe_accuracy")

# ── Multi-model UMAP (all models, same images) ────────────────────────────────
# Stack representations from all models for the same images into one UMAP,
# coloured by model identity — shows how different models structure the space.
MULTI_SAMPLE = 200  # per model (keep total manageable)
if len(mc_repr) >= 3 and common_mc:
    multi_ids = sorted(rng.choice(sorted(common_mc),
                                   min(MULTI_SAMPLE, len(common_mc)), replace=False))
    all_reps_list  = []
    all_model_labels = []
    all_class_labels = []

    for m in cka_models_valid if cka_mat is not None else mc_models[:5]:
        raws = load_final_reps(m, mc_repr[m], multi_ids)
        for img_id, r in zip(multi_ids, raws):
            if r is None:
                continue
            all_reps_list.append(r)
            all_model_labels.append(DISPLAY[m])
            fname = img_id + ".png"
            all_class_labels.append(gt.get(fname, {}).get("anomaly_class", "normal"))

    if all_reps_list:
        # Project each model's reps to 128-d PCA before stacking
        # (models have different hidden dims — e.g. 3584 vs 896)
        from sklearn.decomposition import PCA
        from collections import defaultdict as _dd
        model_buckets = _dd(list)
        model_bucket_idx = _dd(list)
        for idx_, (m_, r_) in enumerate(zip(all_model_labels, all_reps_list)):
            model_buckets[m_].append(r_)
            model_bucket_idx[m_].append(idx_)
        proj_reps = [None] * len(all_reps_list)
        pca_dim = 128
        for m_, bucket in model_buckets.items():
            X_ = np.stack(bucket)
            n_comp = min(pca_dim, X_.shape[0] - 1, X_.shape[1])
            pca_ = PCA(n_components=n_comp, random_state=42)
            X_p  = pca_.fit_transform(X_)
            # pad to pca_dim if needed
            if X_p.shape[1] < pca_dim:
                X_p = np.pad(X_p, ((0,0),(0, pca_dim - X_p.shape[1])))
            for orig_idx, row in zip(model_bucket_idx[m_], X_p):
                proj_reps[orig_idx] = row
        all_reps_list = proj_reps
        multi_mat = np.stack(all_reps_list)
        try:
            import umap as umap_lib
            reducer2 = umap_lib.UMAP(n_components=2, n_neighbors=20, min_dist=0.15,
                                     random_state=42, metric="cosine", n_jobs=-1)
            emb2 = reducer2.fit_transform(multi_mat)
        except ImportError:
            from sklearn.decomposition import PCA
            emb2 = PCA(n_components=2, random_state=42).fit_transform(multi_mat)

        unique_models = sorted(set(all_model_labels))
        model_cmap    = plt.cm.get_cmap("tab10", len(unique_models))
        model_to_idx  = {m: i for i, m in enumerate(unique_models)}

        fig, ax = plt.subplots(figsize=(8, 6))
        for mdl in unique_models:
            mask = np.array([l == mdl for l in all_model_labels])
            ax.scatter(emb2[mask, 0], emb2[mask, 1],
                       s=10, alpha=0.55, color=model_cmap(model_to_idx[mdl]),
                       label=mdl, linewidths=0)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title("Representation Space Across Models\n(Anomalous Images)", fontsize=10)
        ax.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.02, 1),
                  loc="upper left", borderaxespad=0)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"umap_by_model.{ext}")
        plt.close(fig)
        print("Saved umap_by_model")

# ── Save summary JSON ─────────────────────────────────────────────────────────
repr_summary = {
    "umap_model":     umap_model,
    "n_umap_images":  len(umap_ids_valid) if umap_ids_valid else 0,
    "cka":            cka_summary,
    "probe_model":    probe_model,
    "probe_n_layers": n_layers if "n_layers" in dir() else None,
    "probe_accs":     [round(a, 4) for a in probe_accs] if "probe_accs" in dir() else [],
}
with open(OUT_DIR / "representation_summary.json", "w") as f:
    json.dump(repr_summary, f, indent=2)
print("Saved representation_summary.json")

print(f"\nAll representation outputs saved to: {OUT_DIR}")
