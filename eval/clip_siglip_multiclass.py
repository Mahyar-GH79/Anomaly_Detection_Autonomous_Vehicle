"""
Zero-Shot CLIP + SigLIP Multiclass Anomaly Classification
==========================================================
Runs all CLIP and SigLIP variants on the multiclass anomaly classification
task (only the ~5,000 anomalous images, classifying into 11 anomaly classes).

For each variant:
  1. Encodes 11 class-description text prompts
  2. For each anomalous image, computes cosine similarity with each prompt
  3. Predicts the class with the highest similarity
  4. Saves per-image predictions and aggregate metrics

Output structure:
  Baselines/Outputs/clip_predictions_multiclass/{variant}/
    predictions.json
    metrics.json
    confusion_matrix.pdf/png
    per_class_f1.pdf/png
  Baselines/Outputs/siglip_predictions_multiclass/{variant}/
    (same files)

Usage:
  CUDA_VISIBLE_DEVICES=1 python clip_siglip_multiclass.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from tqdm import tqdm
from transformers import (
    AutoModel, AutoProcessor,
    CLIPModel, CLIPProcessor,
    SiglipModel, SiglipProcessor,
)

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

CLIP_VARIANTS = [
    ("vit_b32", "openai/clip-vit-base-patch32"),
    ("vit_b16", "openai/clip-vit-base-patch16"),
    ("vit_l14", "openai/clip-vit-large-patch14"),
]

SIGLIP_VARIANTS = [
    ("base_patch16_224",   "google/siglip-base-patch16-224"),
    ("large_patch16_256",  "google/siglip-large-patch16-256"),
    ("so400m_patch14_384", "google/siglip-so400m-patch14-384"),
]

# 11 anomaly classes — same as VLM multiclass eval
ANOMALY_CLASSES = [
    "animal_on_road",
    "extreme_weather",
    "road_surface_hazard",
    "fallen_debris_or_vegetation",
    "strange_object_on_road",
    "vehicle_incident",
    "infrastructure_failure",
    "human_presence_anomaly",
    "adverse_lighting",
    "oversized_or_unusual_vehicle",
    "multi_hazard_compound",
]

# Natural-language descriptions for each class — designed so CLIP/SigLIP
# can match visual content effectively.
CLASS_PROMPTS = {
    "animal_on_road":
        "a photo of an animal standing on or crossing a road in front of a vehicle",
    "extreme_weather":
        "a photo of a road in extreme weather conditions like heavy fog, flooding, or a blizzard",
    "road_surface_hazard":
        "a photo of a damaged road with potholes, sinkholes, or surface debris",
    "fallen_debris_or_vegetation":
        "a photo of fallen tree branches, leaves, or debris blocking a road",
    "strange_object_on_road":
        "a photo of an unusual object lying on a road such as furniture or a tire",
    "vehicle_incident":
        "a photo of a vehicle accident, crash, or collision on a road",
    "infrastructure_failure":
        "a photo of damaged road infrastructure like fallen traffic signs, broken barriers, or downed utility poles",
    "human_presence_anomaly":
        "a photo of a pedestrian or person walking on a highway or in an unsafe road area",
    "adverse_lighting":
        "a photo of a road scene with extremely poor lighting or strong glare from headlights or sun",
    "oversized_or_unusual_vehicle":
        "a photo of an oversized or unusual vehicle such as a large truck, construction vehicle, or trailer",
    "multi_hazard_compound":
        "a photo of a road scene with multiple simultaneous hazards or dangerous conditions",
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

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def load_ground_truth(dataset_json: Path):
    """Returns {filename: {anomaly_present, anomaly_class}} for anomalous samples."""
    with open(dataset_json) as f:
        data = json.load(f)
    samples = data.get("samples", data)
    gt = {}
    for fname, rec in samples.items():
        if not isinstance(rec, dict):
            continue
        if not rec.get("anomaly_present"):
            continue
        ac = rec.get("anomaly_class")
        if ac in ANOMALY_CLASSES:
            gt[fname] = {"anomaly_class": ac}
    return gt


def to_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Cannot extract tensor from {type(output)}")


def load_model(model_id: str, family: str, device):
    print(f"[INFO] Loading {family.upper()}: {model_id} ...")
    if family == "clip":
        try:
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id).to(device)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id).to(device)
    else:  # siglip
        try:
            processor = SiglipProcessor.from_pretrained(model_id)
            model = SiglipModel.from_pretrained(model_id).to(device)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def encode_texts(model, processor, prompts, device):
    inputs = processor(text=prompts, return_tensors="pt", padding=True,
                       truncation=True).to(device)
    feats  = to_tensor(model.get_text_features(**inputs))
    feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats   # (n_classes, d)


@torch.no_grad()
def encode_image(model, processor, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    feats  = to_tensor(model.get_image_features(**inputs))
    feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats   # (1, d)


def plot_confusion(cm_norm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([CLASS_DISPLAY.get(l, l) for l in labels],
                        rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([CLASS_DISPLAY.get(l, l) for l in labels], fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            c = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=c)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Normalised count", shrink=0.8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(str(out_path) + f".{ext}")
    plt.close(fig)


def plot_per_class_f1(per_class, out_path, title):
    classes = list(per_class.keys())
    f1s     = [per_class[c]["f1-score"] for c in classes]
    support = [int(per_class[c]["support"]) for c in classes]

    order = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.array([f1s[i] for i in order]))
    bars = ax.barh([CLASS_DISPLAY.get(classes[i], classes[i]) for i in order],
                   [f1s[i] for i in order],
                   color=colors, edgecolor="gray", linewidth=0.5)
    for bar, val, n in zip(bars, [f1s[i] for i in order], [support[i] for i in order]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} (n={n})", va="center", fontsize=7, color="gray")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-Score")
    ax.set_title(title)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(str(out_path) + f".{ext}")
    plt.close(fig)


def run_variant(variant_name, model_id, family,
                images_dir, gt_map, out_dir, device):
    out_subdir = out_dir / variant_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    metrics_file = out_subdir / "metrics.json"
    if metrics_file.exists():
        print(f"[SKIP] {variant_name} — metrics.json already exists")
        return

    model, processor = load_model(model_id, family, device)

    # Encode text prompts (in fixed class order)
    prompts = [CLASS_PROMPTS[c] for c in ANOMALY_CLASSES]
    text_feats = encode_texts(model, processor, prompts, device)   # (11, d)

    # Iterate images
    image_paths = [images_dir / fname for fname in sorted(gt_map.keys())]
    image_paths = [p for p in image_paths if p.exists()]
    print(f"[INFO] Processing {len(image_paths)} anomalous images")

    results = []
    skipped = 0

    for img_path in tqdm(image_paths, desc=f"{family}/{variant_name}", unit="img"):
        fname = img_path.name
        gt    = gt_map[fname]["anomaly_class"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            skipped += 1
            continue

        img_feats = encode_image(model, processor, img, device)        # (1, d)
        sims      = (img_feats @ text_feats.T).squeeze().cpu().numpy() # (11,)
        # Softmax over classes (cosine sims are already in [-1,1])
        probs = np.exp(sims - sims.max())
        probs = probs / probs.sum()
        pred_idx = int(np.argmax(probs))
        pred_class = ANOMALY_CLASSES[pred_idx]

        results.append({
            "filename":            fname,
            "ground_truth_class":  gt,
            "predicted_class":     pred_class,
            "scores":              {ANOMALY_CLASSES[i]: round(float(probs[i]), 6)
                                    for i in range(len(ANOMALY_CLASSES))},
            "correct":             pred_class == gt,
        })

    print(f"[INFO] Processed: {len(results)}  |  Skipped: {skipped}")

    # ── Metrics ────────────────────────────────────────────────────────────────
    y_true = [r["ground_truth_class"] for r in results]
    y_pred = [r["predicted_class"]    for r in results]

    accuracy        = accuracy_score(y_true, y_pred)
    bal_acc         = balanced_accuracy_score(y_true, y_pred)
    macro_f1        = f1_score(y_true, y_pred, average="macro",     zero_division=0)
    weighted_f1     = f1_score(y_true, y_pred, average="weighted",  zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall    = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    kappa           = cohen_kappa_score(y_true, y_pred)

    cls_report = classification_report(y_true, y_pred, output_dict=True,
                                       zero_division=0,
                                       labels=ANOMALY_CLASSES)
    per_class = {c: cls_report[c] for c in ANOMALY_CLASSES if c in cls_report}

    cm        = confusion_matrix(y_true, y_pred, labels=ANOMALY_CLASSES)
    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    metrics = {
        "model":             model_id,
        "variant":           variant_name,
        "family":            family,
        "n_evaluated":       len(results),
        "accuracy":          round(accuracy,        4),
        "balanced_accuracy": round(bal_acc,         4),
        "macro_f1":          round(macro_f1,        4),
        "weighted_f1":       round(weighted_f1,     4),
        "macro_precision":   round(macro_precision, 4),
        "macro_recall":      round(macro_recall,    4),
        "cohen_kappa":       round(kappa,           4),
        "per_class":         {c: {k: round(float(v), 4) for k, v in d.items()}
                              for c, d in per_class.items()},
        "confusion_matrix":  cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "confusion_matrix_labels": ANOMALY_CLASSES,
    }

    # Save predictions
    with open(out_subdir / "predictions.json", "w") as f:
        json.dump({"model": model_id, "results": results}, f, indent=2)

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Figures
    plot_confusion(np.array(cm_norm), ANOMALY_CLASSES,
                   out_subdir / "confusion_matrix",
                   f"{family.upper()} {variant_name} — Multiclass Anomaly")
    plot_per_class_f1(per_class, out_subdir / "per_class_f1",
                      f"{family.upper()} {variant_name} — Per-Class F1")

    print(f"\n  ── {variant_name} ──")
    print(f"    accuracy          : {accuracy:.4f}")
    print(f"    balanced_accuracy : {bal_acc:.4f}")
    print(f"    macro_f1          : {macro_f1:.4f}")
    print(f"    weighted_f1       : {weighted_f1:.4f}")

    # Free memory
    del model, processor, text_feats
    torch.cuda.empty_cache()


def aggregate_results(out_dir, family):
    """Collect metrics across all variants of a family into one summary."""
    rows = []
    for sub in sorted(out_dir.iterdir()):
        mf = sub / "metrics.json"
        if not mf.exists():
            continue
        with open(mf) as f:
            m = json.load(f)
        rows.append({
            "variant":           m["variant"],
            "model":             m["model"],
            "n_evaluated":       m["n_evaluated"],
            "accuracy":          m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1":          m["macro_f1"],
            "weighted_f1":       m["weighted_f1"],
            "cohen_kappa":       m["cohen_kappa"],
        })
    if rows:
        agg_path = out_dir / "all_metrics.json"
        with open(agg_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[INFO] Aggregate {family} metrics → {agg_path}")
        for r in rows:
            print(f"  {r['variant']:<22s}  acc={r['accuracy']:.3f}  "
                  f"bacc={r['balanced_accuracy']:.3f}  "
                  f"macroF1={r['macro_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir",   default=str(ROOT / "Data" / "images"))
    parser.add_argument("--dataset-json", default=str(ROOT / "Data" / "dataset.json"))
    parser.add_argument("--clip-out",     default=str(ROOT / "Baselines" / "Outputs" / "clip_predictions_multiclass"))
    parser.add_argument("--siglip-out",   default=str(ROOT / "Baselines" / "Outputs" / "siglip_predictions_multiclass"))
    parser.add_argument("--family",       choices=["clip", "siglip", "all"], default="all")
    args = parser.parse_args()

    images_dir   = Path(args.images_dir)
    dataset_json = Path(args.dataset_json)
    clip_out     = Path(args.clip_out)
    siglip_out   = Path(args.siglip_out)
    clip_out.mkdir(parents=True, exist_ok=True)
    siglip_out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print(f"[INFO] Loading GT from {dataset_json} ...")
    gt = load_ground_truth(dataset_json)
    print(f"[INFO] Anomalous GT records (with valid class): {len(gt):,}")

    if args.family in ("clip", "all"):
        print(f"\n{'='*60}\n  CLIP variants\n{'='*60}")
        for variant_name, model_id in CLIP_VARIANTS:
            run_variant(variant_name, model_id, "clip",
                        images_dir, gt, clip_out, device)
        aggregate_results(clip_out, "clip")

    if args.family in ("siglip", "all"):
        print(f"\n{'='*60}\n  SigLIP variants\n{'='*60}")
        for variant_name, model_id in SIGLIP_VARIANTS:
            run_variant(variant_name, model_id, "siglip",
                        images_dir, gt, siglip_out, device)
        aggregate_results(siglip_out, "siglip")


if __name__ == "__main__":
    main()
