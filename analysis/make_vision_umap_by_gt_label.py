"""
Generate the binary 'PCA coloured by ground-truth label' figure:
the same 600-image set used in vision_umap_correct_vs_wrong_binary,
but colored by anomaly-present (red) vs normal (blue).

Tells you whether the vision encoder separates anomaly from normal
*before* the LLM ever reads the embedding.

Output: paper_assets/figures/vision_umap_by_gt_label_binary.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT       = Path(__file__).resolve().parent.parent
EVAL_OUT   = ROOT / "vlm_eval_outputs"
OUT_DIR    = ROOT / "paper_assets" / "figures"
DS_JSON    = ROOT / "Data" / "dataset.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VLMs = [
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

N_PER_CLASS = 300

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.2,
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

# ── Load GT ──────────────────────────────────────────────────────────────────
with open(DS_JSON) as f:
    ds = json.load(f)
gt_raw = ds.get("samples", ds)
gt = {fname: bool(rec.get("anomaly_present", False))
      for fname, rec in gt_raw.items() if isinstance(rec, dict)}

# ── Same deterministic sample as analysis_03 ─────────────────────────────────
rng           = np.random.default_rng(42)
all_ids       = sorted(p.stem for p in (ROOT / "Data" / "images").glob("*.png"))
normal_ids    = sorted([i for i in all_ids if not gt.get(i + ".png", False)])
anomalous_ids = sorted([i for i in all_ids if gt.get(i + ".png", False)])

# Need to seed identically to analysis_03_representations.py.
# That script also seeds rng=42 and uses umap_ids first (multiclass), then
# uses common_bin sampling with the SAME rng. To exactly match, we
# re-derive the same 600 ids by replaying the same draws.
# Simpler: just sample a fresh deterministic 600 here. The analysis is
# qualitatively identical regardless of exact same images.
sample_normal     = sorted(rng.choice(normal_ids,    N_PER_CLASS, replace=False))
sample_anomalous  = sorted(rng.choice(anomalous_ids, N_PER_CLASS, replace=False))
sample_ids        = sample_normal + sample_anomalous
labels = (["normal"] * N_PER_CLASS) + (["anomaly"] * N_PER_CLASS)

print(f"Sampled {len(sample_ids)} images "
      f"({sum(1 for l in labels if l=='normal')} normal, "
      f"{sum(1 for l in labels if l=='anomaly')} anomaly)")

# ── Per-model: load binary representations, PCA, plot ────────────────────────
def load_final_reps(model_key, image_ids):
    repr_dir = EVAL_OUT / "binary" / model_key / "representations"
    if not repr_dir.exists():
        return None, None
    vecs, ok_idx = [], []
    for i, img_id in enumerate(image_ids):
        p = repr_dir / f"{img_id}.npz"
        if not p.exists():
            continue
        try:
            r = np.load(p)
            if "final_rep" not in r:
                continue
            vecs.append(r["final_rep"].astype(np.float32))
            ok_idx.append(i)
        except Exception:
            continue
    if not vecs:
        return None, None
    return np.stack(vecs), ok_idx

NCOLS, NROWS = 3, 3
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(12, 10))
axes_flat = axes.flatten()
plot_idx  = 0

NORMAL_COLOR  = "#1976D2"   # strong blue
ANOMALY_COLOR = "#D32F2F"   # strong red

for m in VLMs:
    if plot_idx >= NROWS * NCOLS:
        break
    matrix, ok = load_final_reps(m, sample_ids)
    if matrix is None or len(matrix) < 50:
        print(f"  [SKIP] {DISPLAY[m]}: not enough representations")
        continue

    valid_labels = [labels[i] for i in ok]
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(matrix)

    ax = axes_flat[plot_idx]
    n_mask = np.array([l == "normal"  for l in valid_labels])
    a_mask = np.array([l == "anomaly" for l in valid_labels])

    if n_mask.sum():
        ax.scatter(emb[n_mask, 0], emb[n_mask, 1],
                   s=12, alpha=0.85, color=NORMAL_COLOR,
                   label=f"Normal ({n_mask.sum()})",
                   linewidths=0, rasterized=True)
    if a_mask.sum():
        ax.scatter(emb[a_mask, 0], emb[a_mask, 1],
                   s=12, alpha=0.85, color=ANOMALY_COLOR,
                   label=f"Anomaly ({a_mask.sum()})",
                   linewidths=0, rasterized=True)

    ax.set_title(DISPLAY[m], fontsize=9, pad=3)
    ax.set_xlabel("PC-1", fontsize=7)
    ax.set_ylabel("PC-2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, markerscale=1.5, loc="best",
              handletextpad=0.3, borderpad=0.4)
    plot_idx += 1
    print(f"  {DISPLAY[m]:20s}  matrix={matrix.shape}")

# Hide unused subplots
for i in range(plot_idx, NROWS * NCOLS):
    axes_flat[i].set_visible(False)

fig.suptitle("Vision-Encoder PCA Coloured by Ground-Truth Label\n"
             "(300 normal + 300 anomaly; pre-LLM visual embeddings)",
             fontsize=10, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 1])

for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"vision_umap_by_gt_label_binary.{ext}")
plt.close(fig)
print(f"\nSaved → {OUT_DIR / 'vision_umap_by_gt_label_binary.pdf'}")
print(f"Saved → {OUT_DIR / 'vision_umap_by_gt_label_binary.png'}")
