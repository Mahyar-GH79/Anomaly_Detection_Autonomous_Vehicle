"""
Analysis 4 – LLM Layer Hidden-State PCA
=========================================
For each model, runs a forward pass on 600 fixed images and captures the
**last-token hidden state** at every LLM transformer layer via forward hooks.
Then plots PCA of those hidden states for representative layers.

Two tasks:
  binary    – 300 normal + 300 anomalous; colour = normal vs anomalous
  multiclass – 600 anomalous (balanced across classes); colour = anomaly class

Output structure:
  vlm_eval_outputs/analysis/llm_layers/
    binary/
      {model_key}_binary_layer_pca.pdf/png   ← 2×4 grid, 8 representative layers
    multiclass/
      {model_key}_multiclass_layer_pca.pdf/png

Run:
  CUDA_VISIBLE_DEVICES=0 python analysis_04_llm_layers.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.2,
    "grid.linestyle":    "--",
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

ROOT      = Path(__file__).resolve().parent.parent / "vlm_eval_outputs"
OUT_ROOT  = ROOT / "analysis" / "llm_layers"
IMG_DIR   = Path(__file__).resolve().parent.parent / "Data" / "images"
DS_JSON   = Path(__file__).resolve().parent.parent / "Data" / "dataset.json"

(OUT_ROOT / "binary").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "multiclass").mkdir(parents=True, exist_ok=True)

N_SAMPLE     = 600    # total images per task
N_REPR_LAYERS = 8     # how many representative layers to show per plot
PLOT_COLS    = 4
PLOT_ROWS    = 2      # 2 × 4 = 8 subplots

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
MODEL_TO_LOADER = {
    "internvl3_1b":       "internvl3",
    "internvl3_2b":       "internvl3",
    "internvl3_8b":       "internvl3",
    "qwen2vl_2b":         "qwen2vl",
    "qwen25vl_3b":        "qwen25vl",
    "qwen25vl_7b":        "qwen25vl",
    "llava_onevision_7b": "llava_onevision",
    "llava_13b":          "llava_next",
    "llama32_11b":        "llama32",
}
MODEL_HF_IDS = {
    "internvl3_1b":       "OpenGVLab/InternVL3-1B-hf",
    "internvl3_2b":       "OpenGVLab/InternVL3-2B-hf",
    "internvl3_8b":       "OpenGVLab/InternVL3-8B-hf",
    "qwen2vl_2b":         "Qwen/Qwen2-VL-2B-Instruct",
    "qwen25vl_3b":        "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen25vl_7b":        "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava_onevision_7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava_13b":          "llava-hf/llava-v1.6-vicuna-13b-hf",
    "llama32_11b":        "meta-llama/Llama-3.2-11B-Vision-Instruct",
}
CLASS_DISPLAY = {
    "animal_on_road":              "Animal",
    "extreme_weather":             "Ext. Weather",
    "road_surface_hazard":         "Road Hazard",
    "fallen_debris_or_vegetation": "Debris/Veg.",
    "strange_object_on_road":      "Strange Obj.",
    "vehicle_incident":            "Vehicle Inc.",
    "infrastructure_failure":      "Infra. Fail.",
    "human_presence_anomaly":      "Human Pres.",
    "adverse_lighting":            "Adv. Light.",
    "oversized_or_unusual_vehicle":"Oversized Veh.",
    "multi_hazard_compound":       "Multi-Hazard",
    "normal":                      "Normal",
}

# ── Import LOADERS and prompts from vlm_eval_tasks.py ─────────────────────────
spec = importlib.util.spec_from_file_location(
    "vlm_eval_tasks", Path(__file__).resolve().parent.parent / "eval" / "vlm_eval_tasks.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["vlm_eval_tasks"] = mod
spec.loader.exec_module(mod)

LOADERS         = mod.LOADERS
SYSTEM_PROMPT   = mod.SYSTEM_PROMPT
BINARY_PROMPT   = mod.BINARY_PROMPT
MULTICLASS_PROMPT = mod.MULTICLASS_PROMPT

# ── Load ground truth ─────────────────────────────────────────────────────────
with open(DS_JSON) as f:
    ds = json.load(f)
gt_raw = ds.get("samples", ds)
gt = {}
for fname, rec in gt_raw.items():
    if not isinstance(rec, dict):
        continue
    gt[fname] = {
        "anomaly_present": rec.get("anomaly_present", False),
        "anomaly_class":   rec.get("anomaly_class", "normal") if rec.get("anomaly_present") else "normal",
    }

all_fnames  = sorted(gt.keys())
normal_fnames     = [f for f in all_fnames if not gt[f]["anomaly_present"]]
anomalous_fnames  = [f for f in all_fnames if gt[f]["anomaly_present"]]

# ── Fixed 600-image sample sets (same across all models) ─────────────────────
rng = np.random.default_rng(42)

# Binary: 300 normal + 300 anomalous
bin_normal   = list(rng.choice(normal_fnames,    300, replace=False))
bin_anomalous = list(rng.choice(anomalous_fnames, 300, replace=False))
binary_fnames = bin_normal + bin_anomalous
binary_labels = (["normal"] * 300) + (["anomalous"] * 300)

# Multiclass: 600 anomalous, balanced across classes as much as possible
class_groups = {}
for f in anomalous_fnames:
    c = gt[f]["anomaly_class"]
    class_groups.setdefault(c, []).append(f)

mc_fnames = []
classes_sorted = sorted(class_groups.keys())
n_classes = len(classes_sorted)
per_class = max(1, N_SAMPLE // n_classes)
for c in classes_sorted:
    pool = class_groups[c]
    n    = min(per_class, len(pool))
    mc_fnames.extend(rng.choice(pool, n, replace=False).tolist())
# top-up if < 600
remaining = [f for f in anomalous_fnames if f not in set(mc_fnames)]
if len(mc_fnames) < N_SAMPLE and remaining:
    top_up = rng.choice(remaining,
                        min(N_SAMPLE - len(mc_fnames), len(remaining)),
                        replace=False).tolist()
    mc_fnames.extend(top_up)
mc_fnames  = mc_fnames[:N_SAMPLE]
mc_labels  = [gt[f]["anomaly_class"] for f in mc_fnames]

print(f"Binary sample   : {len(binary_fnames)} images "
      f"({sum(1 for l in binary_labels if l=='normal')} normal, "
      f"{sum(1 for l in binary_labels if l=='anomalous')} anomalous)")
print(f"Multiclass sample: {len(mc_fnames)} images across "
      f"{len(set(mc_labels))} classes")

# ── LLM layer finder ─────────────────────────────────────────────────────────
LLM_LAYER_PATHS = [
    "language_model.model.layers",        # InternVL3, LLaVA variants
    "model.language_model.model.layers",
    "model.layers",                        # Qwen2-VL, Qwen2.5-VL
    "model.model.layers",                  # some LLaMA wrappers
    "language_model.model.layers",
    "model.vision_model.encoder.layers",   # fallback (shouldn't hit)
]

def find_llm_layers(model):
    for path in LLM_LAYER_PATHS:
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 4:
                print(f"  Found LLM layers at '{path}' ({len(obj)} layers)")
                return list(obj), len(obj)
        except AttributeError:
            continue
    # Generic fallback: largest ModuleList that is NOT a vision component
    best = ([], 0)
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList):
            continue
        if len(module) <= best[1]:
            continue
        if any(kw in name.lower() for kw in ("vision", "visual", "vit", "patch", "img")):
            continue
        best = (list(module), len(module))
    if best[1] > 4:
        print(f"  Found LLM layers via fallback ({best[1]} layers)")
        return best
    print("  [WARN] Could not find LLM layers")
    return [], 0

def pick_representative_layers(n_layers, n_pick=N_REPR_LAYERS):
    """Pick n_pick evenly-spaced layer indices from 1 to n_layers-1.
    Layer 0 is skipped: its last-token hidden state is the raw token embedding
    of the prompt's final token, identical across all images → degenerate PCA."""
    if n_layers <= 1:
        return list(range(n_layers))
    start = min(1, n_layers - 1)
    if n_layers - start <= n_pick:
        return list(range(start, n_layers))
    indices = np.round(np.linspace(start, n_layers - 1, n_pick)).astype(int)
    return indices.tolist()

# ── PCA helper ────────────────────────────────────────────────────────────────
def pca2d(X):
    """Simple 2-component PCA, returns (n,2) float32."""
    from sklearn.decomposition import PCA
    X = X.astype(np.float32)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)

# ── Collect hidden states for one model ───────────────────────────────────────
def collect_hidden_states(model, processor, infer_fn,
                          fnames, system_prompt, user_prompt):
    """
    Runs inference on each image in fnames.
    Captures last-token hidden state at every LLM layer during the prefill pass.

    Returns:
        layer_reps : dict  layer_idx → np.array (n_images, hidden_dim)
        valid_mask : list[bool]  True if inference succeeded
    """
    llm_layers, n_layers = find_llm_layers(model)
    if n_layers == 0:
        return {}, [False] * len(fnames)

    # Storage: one list per layer
    captured   = {i: [] for i in range(n_layers)}
    valid_mask = []

    # Register hooks — capture last token only during prefill (seq_len > 1)
    def make_hook(layer_idx):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if h.shape[1] > 1:   # prefill pass only
                vec = h[0, -1, :].detach().cpu().float().numpy()
                captured[layer_idx].append(vec)
        return hook

    hooks = [layer.register_forward_hook(make_hook(i))
             for i, layer in enumerate(llm_layers)]

    for idx, fname in enumerate(fnames):
        img_path = IMG_DIR / fname
        if not img_path.exists():
            valid_mask.append(False)
            # push a sentinel so indices stay aligned
            for i in range(n_layers):
                captured[i].append(None)
            continue

        img = Image.open(img_path).convert("RGB")
        try:
            with torch.no_grad():
                infer_fn(model, processor, img, system_prompt, user_prompt)
            valid_mask.append(True)
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")
            valid_mask.append(False)
            for i in range(n_layers):
                # If hook didn't fire, append None
                if len(captured[i]) < idx + 1:
                    captured[i].append(None)

        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{len(fnames)} done")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Stack into arrays, skipping failed images
    layer_reps = {}
    for i in range(n_layers):
        vecs = [v for v in captured[i] if v is not None]
        if vecs:
            layer_reps[i] = np.stack(vecs)

    # Rebuild valid_mask aligned to captured (some hooks may fire extra)
    return layer_reps, valid_mask, n_layers

# ── Plotting function ──────────────────────────────────────────────────────────
def plot_layer_pca(layer_reps, rep_layer_indices, labels, label_colors,
                   label_names, valid_mask, model_key, task, n_layers_total):
    """
    layer_reps        : dict  layer_idx → (n_valid, dim)
    rep_layer_indices : list of layer indices to plot
    labels            : list of str labels per image (full, pre-filter)
    label_colors      : dict  label_str → hex color
    label_names       : dict  label_str → display name
    valid_mask        : list[bool]
    """
    labels_valid = [l for l, ok in zip(labels, valid_mask) if ok]

    fig, axes = plt.subplots(PLOT_ROWS, PLOT_COLS,
                             figsize=(PLOT_COLS * 3.2, PLOT_ROWS * 3.0))
    axes_flat = axes.flatten()

    unique_labels = sorted(set(labels_valid))

    for plot_i, layer_idx in enumerate(rep_layer_indices):
        ax  = axes_flat[plot_i]
        mat = layer_reps.get(layer_idx)
        if mat is None or mat.shape[0] < 10:
            ax.set_visible(False)
            continue

        emb = pca2d(mat)

        # Detect *truly* degenerate PCA only (e.g. layer 0 with identical
        # embeddings). Lower threshold so meaningful but tight clusters still
        # render — earlier 1e-4 was too aggressive for LLaMA-3.2-Vision layer 1.
        spread = emb.std(axis=0).max()
        if spread < 1e-7:
            ax.text(0.5, 0.5, "Degenerate\n(all reps identical)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="gray",
                    bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))
            ax.set_title(f"Layer {layer_idx} / {n_layers_total - 1}", fontsize=8, pad=3)
            ax.set_xlabel("PC-1", fontsize=7)
            ax.set_ylabel("PC-2", fontsize=7)
            continue

        for lbl in unique_labels:
            mask = np.array([l == lbl for l in labels_valid])
            if not mask.any():
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       s=14, alpha=0.85,
                       color=label_colors.get(lbl, "#888888"),
                       label=label_names.get(lbl, lbl),
                       linewidths=0, rasterized=True)

        ax.set_title(f"Layer {layer_idx} / {n_layers_total - 1}", fontsize=8, pad=3)
        ax.set_xlabel("PC-1", fontsize=7)
        ax.set_ylabel("PC-2", fontsize=7)
        ax.tick_params(labelsize=5)

        if plot_i == 0:
            ax.legend(fontsize=6, markerscale=1.5,
                      handletextpad=0.3, borderpad=0.4,
                      loc="best", ncol=1)

    # Hide any unused subplots
    for i in range(len(rep_layer_indices), PLOT_ROWS * PLOT_COLS):
        axes_flat[i].set_visible(False)

    task_title = ("Binary (Normal vs Anomalous)"
                  if task == "binary"
                  else "Multiclass (Anomaly Type)")
    fig.suptitle(
        f"{DISPLAY[model_key]} — LLM Layer Hidden-State PCA\n{task_title}",
        fontsize=10, y=1.02)
    plt.tight_layout()

    out_dir = OUT_ROOT / task
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{model_key}_{task}_layer_pca.{ext}")
    plt.close(fig)
    print(f"  Saved {model_key}_{task}_layer_pca")

# ── Colour maps ───────────────────────────────────────────────────────────────
BINARY_COLORS = {
    "normal":    "#1565C0",   # deep blue
    "anomalous": "#C62828",   # deep red
}
BINARY_NAMES  = {"normal": "Normal", "anomalous": "Anomalous"}

all_mc_classes = sorted(set(mc_labels))
# Bold, saturated palette — 11 distinct colours
BOLD_PALETTE = [
    "#E63946",  # vivid red
    "#1976D2",  # strong blue
    "#43A047",  # vivid green
    "#FB8C00",  # bright orange
    "#8E24AA",  # purple
    "#00ACC1",  # cyan
    "#D81B60",  # magenta
    "#7CB342",  # lime
    "#3949AB",  # indigo
    "#F4511E",  # deep orange
    "#5D4037",  # dark brown
]
MC_COLORS = {c: BOLD_PALETTE[i % len(BOLD_PALETTE)]
             for i, c in enumerate(all_mc_classes)}
MC_NAMES  = {c: CLASS_DISPLAY.get(c, c) for c in all_mc_classes}

# ── Main loop: one model at a time ────────────────────────────────────────────
for model_key in MODELS:
    loader_key = MODEL_TO_LOADER[model_key]
    hf_id      = MODEL_HF_IDS[model_key]

    # Check if already done
    bin_done = (OUT_ROOT / "binary"     / f"{model_key}_binary_layer_pca.pdf").exists()
    mc_done  = (OUT_ROOT / "multiclass" / f"{model_key}_multiclass_layer_pca.pdf").exists()
    if bin_done and mc_done:
        print(f"[SKIP] {DISPLAY[model_key]} — both tasks already done")
        continue

    print(f"\n{'='*60}")
    print(f"Loading {DISPLAY[model_key]} ({hf_id}) ...")
    print(f"{'='*60}")

    try:
        model, processor, infer_fn = LOADERS[loader_key](hf_id, use_4bit=False)
    except Exception as e:
        print(f"[ERROR] Could not load {model_key}: {e}")
        continue

    # ── Binary task ───────────────────────────────────────────────────────────
    if not bin_done:
        print(f"\n  [Binary] Collecting hidden states for {len(binary_fnames)} images...")
        layer_reps_bin, valid_bin, n_layers = collect_hidden_states(
            model, processor, infer_fn,
            binary_fnames, SYSTEM_PROMPT, BINARY_PROMPT)

        rep_layers = pick_representative_layers(n_layers, N_REPR_LAYERS)
        print(f"  n_layers={n_layers}, representative={rep_layers}")

        plot_layer_pca(layer_reps_bin, rep_layers,
                       binary_labels, BINARY_COLORS, BINARY_NAMES,
                       valid_bin, model_key, "binary", n_layers)
        del layer_reps_bin

    # ── Multiclass task ───────────────────────────────────────────────────────
    if not mc_done:
        print(f"\n  [Multiclass] Collecting hidden states for {len(mc_fnames)} images...")
        layer_reps_mc, valid_mc, n_layers = collect_hidden_states(
            model, processor, infer_fn,
            mc_fnames, SYSTEM_PROMPT, MULTICLASS_PROMPT)

        rep_layers = pick_representative_layers(n_layers, N_REPR_LAYERS)
        plot_layer_pca(layer_reps_mc, rep_layers,
                       mc_labels, MC_COLORS, MC_NAMES,
                       valid_mc, model_key, "multiclass", n_layers)
        del layer_reps_mc

    # ── Unload model to free VRAM ─────────────────────────────────────────────
    print(f"\n  Unloading {DISPLAY[model_key]}...")
    del model, processor, infer_fn
    torch.cuda.empty_cache()
    import gc; gc.collect()

print("\nAll LLM layer PCA figures saved to:", OUT_ROOT)
