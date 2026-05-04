"""
Zero-Shot CLIP Anomaly Detection
=================================
Classifies every image in Data/images/ as normal or anomalous using
zero-shot CLIP with two fixed text prompts:
  - "A normal driving scene"
  - "A anomalous driving scene"

The predicted label (normal / anomalous) and raw softmax scores are saved
alongside the ground truth from dataset.json so classification performance
can be computed directly.

Outputs (all written to --output-dir):
  clip_predictions.json   per-image results (scores, predicted label, gt label)
  clip_predictions.csv    same data in tabular form
  clip_metrics.json       accuracy, precision, recall, F1, AUROC, AUPRC

Usage:
    python clip_zero_shot.py \
        --images-dir  ./Data/images \
        --dataset-json ./Data/dataset.json \
        --output-dir  ./clip_predictions

    # use a larger CLIP backbone
    python clip_zero_shot.py --model openai/clip-vit-large-patch14

Requirements:
    pip install transformers torch torchvision Pillow tqdm scikit-learn
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import CLIPModel, CLIPProcessor
except ImportError as e:
    sys.exit(f"[ERROR] {e}\nInstall: pip install transformers torch torchvision Pillow tqdm")

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )
except ImportError:
    sys.exit("[ERROR] scikit-learn not found.\nInstall: pip install scikit-learn")


# ── Text prompts ──────────────────────────────────────────────────────────────

TEXT_PROMPTS = [
    "A normal driving scene",    # index 0 → label: normal
    "A anomalous driving scene", # index 1 → label: anomalous
]
LABEL_NAMES = ["normal", "anomalous"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ground_truth(dataset_json: Path) -> dict[str, bool]:
    """
    Returns {new_filename: anomaly_present (bool)} for every sample entry.
    Skips the 'metadata' key.
    """
    with open(dataset_json, encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", data)  # handle both top-level and nested
    gt: dict[str, bool] = {}
    for key, record in samples.items():
        if key == "metadata":
            continue
        # record is itself a dict
        if isinstance(record, dict) and "anomaly_present" in record:
            gt[key] = bool(record["anomaly_present"])
    return gt


def load_model(model_name: str, device: torch.device):
    print(f"[INFO] Loading CLIP model: {model_name} ...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def _to_tensor(output) -> torch.Tensor:
    """
    Normalises the return value of get_text/image_features across transformers
    versions.  Older builds return a raw Tensor; newer ones may wrap it in a
    BaseModelOutputWithPooling.  We always want the pooled 2-D tensor.
    """
    if isinstance(output, torch.Tensor):
        return output
    # ModelOutput dataclass — prefer pooler_output (the projected embedding)
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Cannot extract tensor from {type(output)}")


@torch.no_grad()
def encode_texts(model, processor, texts: list[str], device: torch.device) -> torch.Tensor:
    """Returns L2-normalised text feature matrix (n_texts, d)."""
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    features = _to_tensor(model.get_text_features(**inputs))
    features = features / features.norm(dim=-1, keepdim=True)
    return features  # (2, d)


@torch.no_grad()
def encode_image(model, processor, image: Image.Image, device: torch.device) -> torch.Tensor:
    """Returns L2-normalised image feature vector (1, d)."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    features = _to_tensor(model.get_image_features(**inputs))
    features = features / features.norm(dim=-1, keepdim=True)
    return features  # (1, d)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    images_dir: Path,
    dataset_json: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load ground truth
    print(f"[INFO] Loading ground truth from {dataset_json} ...")
    gt_map = load_ground_truth(dataset_json)
    print(f"[INFO] Ground truth entries: {len(gt_map):,}")

    # Collect image paths that have ground truth
    image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    image_paths = [p for p in image_paths if p.name in gt_map]
    print(f"[INFO] Images to process: {len(image_paths):,}")

    if not image_paths:
        sys.exit("[ERROR] No images found that match dataset.json entries. Check --images-dir.")

    # Load model
    model, processor = load_model(model_name, device)

    # Pre-encode text prompts once
    text_features = encode_texts(model, processor, TEXT_PROMPTS, device)  # (2, d)
    logit_scale = model.logit_scale.exp().item()

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = 0

    for img_path in tqdm(image_paths, desc="CLIP inference", unit="img"):
        filename = img_path.name
        gt_label = gt_map[filename]  # True = anomalous

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\n[WARN] Cannot open {filename}: {e}")
            skipped += 1
            continue

        img_features = encode_image(model, processor, image, device)  # (1, d)

        # Cosine similarity scaled by logit_scale, then softmax
        logits = logit_scale * (img_features @ text_features.T)  # (1, 2)
        probs = logits.softmax(dim=-1).squeeze().cpu().tolist()   # [p_normal, p_anomalous]

        pred_idx = int(probs[1] > 0.5)  # 0=normal, 1=anomalous
        pred_label = LABEL_NAMES[pred_idx]

        results.append({
            "filename":         filename,
            "score_normal":     round(probs[0], 6),
            "score_anomalous":  round(probs[1], 6),
            "predicted_label":  pred_label,
            "predicted_anomaly": pred_idx == 1,
            "ground_truth_anomaly": gt_label,
            "correct": (pred_idx == 1) == gt_label,
        })

    print(f"[INFO] Processed: {len(results):,}  |  Skipped: {skipped}")

    # ── Save per-image JSON ──────────────────────────────────────────────────
    predictions_json = output_dir / "clip_predictions.json"
    with open(predictions_json, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "prompts": TEXT_PROMPTS, "results": results}, f, indent=2)
    print(f"[INFO] Predictions saved → {predictions_json}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = output_dir / "clip_predictions.csv"
    fieldnames = ["filename", "score_normal", "score_anomalous",
                  "predicted_label", "predicted_anomaly", "ground_truth_anomaly", "correct"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] CSV saved        → {csv_path}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    y_true  = [int(r["ground_truth_anomaly"]) for r in results]
    y_pred  = [int(r["predicted_anomaly"])    for r in results]
    y_score = [r["score_anomalous"]           for r in results]  # probability of anomaly

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auroc     = roc_auc_score(y_true, y_score)
    auprc     = average_precision_score(y_true, y_score)

    metrics = {
        "model":        model_name,
        "prompts":      TEXT_PROMPTS,
        "total_images": len(results),
        "accuracy":     round(accuracy,  4),
        "precision":    round(precision, 4),
        "recall":       round(recall,    4),
        "f1":           round(f1,        4),
        "auroc":        round(auroc,     4),
        "auprc":        round(auprc,     4),
        "class_distribution": {
            "gt_anomalous": sum(y_true),
            "gt_normal":    len(y_true) - sum(y_true),
            "pred_anomalous": sum(y_pred),
            "pred_normal":    len(y_pred) - sum(y_pred),
        },
    }

    metrics_path = output_dir / "clip_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved    → {metrics_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CLIP Zero-Shot Results ({model_name})")
    print(f"{'='*55}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"  AUPRC     : {auprc:.4f}")
    print(f"{'='*55}")
    print()
    print(classification_report(y_true, y_pred, target_names=["normal", "anomalous"]))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot CLIP anomaly classification on the unified driving dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--images-dir",   "-i", type=str, default="./Data/images",
                        help="Directory containing data_XXXXX.png images (default: ./Data/images)")
    parser.add_argument("--dataset-json", "-d", type=str, default="./Data/dataset.json",
                        help="Path to dataset.json with ground truth (default: ./Data/dataset.json)")
    parser.add_argument("--output-dir",   "-o", type=str, default="./clip_predictions",
                        help="Where to write outputs (default: ./clip_predictions)")
    parser.add_argument("--model",        "-m", type=str, default="openai/clip-vit-base-patch32",
                        help="HuggingFace CLIP model ID (default: openai/clip-vit-base-patch32)")
    parser.add_argument("--batch-size",         type=int, default=1,
                        help="Image batch size (default: 1 — safe for all GPUs)")
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        images_dir=Path(args.images_dir),
        dataset_json=Path(args.dataset_json),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
