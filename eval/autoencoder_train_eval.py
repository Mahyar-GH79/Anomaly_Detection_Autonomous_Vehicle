"""
CNN Autoencoder — Train on Normal, Evaluate for Anomaly Detection
=================================================================
Trains a convolutional autoencoder on normal driving images (autoencoder_train/).
Anomaly detection is done by reconstruction error: anomalous images have higher
MSE because the autoencoder was never trained to reconstruct them.

Evaluation uses the unified Data/images/ dataset with ground truth from
Data/dataset.json. Reports AUROC, AUPRC, and optimal-threshold F1.

Outputs (--output-dir):
  autoencoder.pth             trained model weights
  ae_eval_scores.json         per-image reconstruction MSE + ground truth
  ae_eval_scores.csv          same in tabular form
  ae_metrics.json             AUROC, AUPRC, F1 at optimal threshold

Architecture (CNN autoencoder, input 224×224 RGB):
  Encoder: 224→112→56→28→14→7  (channels: 3→32→64→128→256→512)
  Bottleneck: 7×7×512 spatial feature map (no FC layer — preserves spatial info)
  Decoder: 7→14→28→56→112→224  (channels: 512→256→128→64→32→3)
  Loss: MSE pixel reconstruction

Usage:
    # 1. Train
    python autoencoder_train_eval.py train \
        --train-dir   ./autoencoder_train \
        --output-dir  ./autoencoder_output

    # 2. Evaluate
    python autoencoder_train_eval.py eval \
        --images-dir   ./Data/images \
        --dataset-json ./Data/dataset.json \
        --model-path   ./autoencoder_output/autoencoder.pth \
        --output-dir   ./autoencoder_output

    # 3. Train then evaluate in one shot
    python autoencoder_train_eval.py train_eval \
        --train-dir    ./autoencoder_train \
        --images-dir   ./Data/images \
        --dataset-json ./Data/dataset.json \
        --output-dir   ./autoencoder_output

Requirements:
    pip install torch torchvision Pillow tqdm scikit-learn
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    sys.exit(f"[ERROR] {e}\nInstall: pip install torch torchvision Pillow tqdm")

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )
    import numpy as np
except ImportError:
    sys.exit("[ERROR] scikit-learn / numpy not found.\nInstall: pip install scikit-learn numpy")


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_SIZE   = 224
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS  = 4

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImageFolderFlat(Dataset):
    """Loads all PNG/JPG images in a flat directory. Returns (tensor, path)."""

    def __init__(self, image_dir: Path, transform):
        self.paths = sorted(
            list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        )
        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path.name)


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BatchNorm → LeakyReLU."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """ConvTranspose2d (stride=2 upsample) → BatchNorm → ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class CNNAutoencoder(nn.Module):
    """
    Input: (B, 3, 224, 224)
    Encoder: 224 → 112 → 56 → 28 → 14 → 7
    Decoder: 7 → 14 → 28 → 56 → 112 → 224
    Output: (B, 3, 224, 224), values in any range (loss computed against normalised input)
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  32),
            nn.MaxPool2d(2),  # 224 → 112
        )
        self.enc2 = nn.Sequential(
            ConvBlock(32,  64),
            ConvBlock(64,  64),
            nn.MaxPool2d(2),  # 112 → 56
        )
        self.enc3 = nn.Sequential(
            ConvBlock(64,  128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),  # 56 → 28
        )
        self.enc4 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2),  # 28 → 14
        )
        self.enc5 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(2),  # 14 → 7
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec5 = UpBlock(512, 256)    # 7 → 14
        self.dec4 = nn.Sequential(
            ConvBlock(256, 256),
            UpBlock(256, 128),           # 14 → 28
        )
        self.dec3 = nn.Sequential(
            ConvBlock(128, 128),
            UpBlock(128, 64),            # 28 → 56
        )
        self.dec2 = nn.Sequential(
            ConvBlock(64, 64),
            UpBlock(64, 32),             # 56 → 112
        )
        self.dec1 = nn.Sequential(
            ConvBlock(32, 32),
            UpBlock(32, 16),             # 112 → 224
        )
        self.final = nn.Conv2d(16, 3, kernel_size=1)  # 3-channel output

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        return x

    def decode(self, z):
        z = self.dec5(z)
        z = self.dec4(z)
        z = self.dec3(z)
        z = self.dec2(z)
        z = self.dec1(z)
        z = self.final(z)
        return z

    def forward(self, x):
        return self.decode(self.encode(x))


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    train_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    dataset = ImageFolderFlat(train_dir, TRAIN_TRANSFORM)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[INFO] Training images: {len(dataset):,}  |  Batches/epoch: {len(loader)}")

    model     = CNNAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss   = float("inf")
    train_losses = []

    print(f"\n[INFO] Starting training for {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs}", unit="batch", leave=False)
        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            recon  = model(images)
            loss   = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        train_losses.append(avg_loss)
        print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = output_dir / "autoencoder.pth"
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "loss":       best_loss,
                "image_size": IMAGE_SIZE,
            }, model_path)

    # Save training history
    history_path = output_dir / "train_history.json"
    with open(history_path, "w") as f:
        json.dump({"epochs": list(range(1, epochs + 1)), "loss": train_losses}, f, indent=2)

    print(f"\n[INFO] Best loss : {best_loss:.6f}")
    print(f"[INFO] Model saved → {output_dir / 'autoencoder.pth'}")
    print(f"[INFO] History    → {history_path}")

    return output_dir / "autoencoder.pth"


# ── Evaluation ────────────────────────────────────────────────────────────────

def load_ground_truth(dataset_json: Path) -> dict[str, bool]:
    with open(dataset_json, encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", data)
    gt: dict[str, bool] = {}
    for key, record in samples.items():
        if key == "metadata":
            continue
        if isinstance(record, dict) and "anomaly_present" in record:
            gt[key] = bool(record["anomaly_present"])
    return gt


@torch.no_grad()
def compute_reconstruction_errors(
    model: CNNAutoencoder,
    images_dir: Path,
    gt_map: dict[str, bool],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> list[dict]:
    """
    Runs all images through the autoencoder and returns per-image MSE.
    """
    dataset = ImageFolderFlat(images_dir, EVAL_TRANSFORM)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    criterion = nn.MSELoss(reduction="none")  # keep per-pixel losses
    results = []

    for images, filenames in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device, non_blocking=True)
        recon  = model(images)

        # MSE per image: mean over channels and spatial dims
        per_image_mse = criterion(recon, images).mean(dim=[1, 2, 3]).cpu().tolist()

        for fname, mse in zip(filenames, per_image_mse):
            if fname not in gt_map:
                continue  # image not in ground truth, skip
            results.append({
                "filename":             fname,
                "reconstruction_mse":   round(float(mse), 8),
                "ground_truth_anomaly": gt_map[fname],
            })

    return results


def evaluate(
    images_dir: Path,
    dataset_json: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load model
    print(f"[INFO] Loading model from {model_path} ...")
    checkpoint = torch.load(model_path, map_location=device)
    model = CNNAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"       Trained for {checkpoint.get('epoch', '?')} epochs, "
          f"best loss={checkpoint.get('loss', '?'):.6f}")

    # Load ground truth
    print(f"[INFO] Loading ground truth from {dataset_json} ...")
    gt_map = load_ground_truth(dataset_json)
    print(f"[INFO] Ground truth entries: {len(gt_map):,}")

    # Compute reconstruction errors
    results = compute_reconstruction_errors(
        model, images_dir, gt_map, device, batch_size, num_workers
    )
    print(f"[INFO] Scored {len(results):,} images")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-image scores JSON
    scores_json = output_dir / "ae_eval_scores.json"
    with open(scores_json, "w", encoding="utf-8") as f:
        json.dump({"model_path": str(model_path), "results": results}, f, indent=2)
    print(f"[INFO] Scores saved → {scores_json}")

    # Save CSV
    csv_path = output_dir / "ae_eval_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "reconstruction_mse", "ground_truth_anomaly"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] CSV saved    → {csv_path}")

    # Compute metrics
    y_true  = np.array([int(r["ground_truth_anomaly"]) for r in results])
    y_score = np.array([r["reconstruction_mse"]        for r in results])

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    # Find optimal threshold by maximising F1 over a grid of thresholds
    thresholds = np.percentile(y_score, np.linspace(0, 100, 500))
    best_f1, best_thresh, best_prec, best_rec, best_acc = 0.0, 0.0, 0.0, 0.0, 0.0
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = float(thresh)
            best_prec   = precision_score(y_true, y_pred, zero_division=0)
            best_rec    = recall_score(y_true, y_pred, zero_division=0)
            best_acc    = accuracy_score(y_true, y_pred)

    y_pred_best = (y_score >= best_thresh).astype(int)

    metrics = {
        "model_path": str(model_path),
        "total_images": len(results),
        "auroc":        round(float(auroc),     4),
        "auprc":        round(float(auprc),     4),
        "optimal_threshold": round(best_thresh, 8),
        "at_optimal_threshold": {
            "f1":        round(best_f1,   4),
            "precision": round(best_prec, 4),
            "recall":    round(best_rec,  4),
            "accuracy":  round(best_acc,  4),
        },
        "class_distribution": {
            "gt_anomalous": int(y_true.sum()),
            "gt_normal":    int(len(y_true) - y_true.sum()),
            "pred_anomalous_at_optimal": int(y_pred_best.sum()),
        },
        "reconstruction_mse_stats": {
            "normal_mean":    round(float(y_score[y_true == 0].mean()), 8),
            "normal_std":     round(float(y_score[y_true == 0].std()),  8),
            "anomalous_mean": round(float(y_score[y_true == 1].mean()), 8),
            "anomalous_std":  round(float(y_score[y_true == 1].std()),  8),
        },
    }

    metrics_path = output_dir / "ae_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved → {metrics_path}")

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Autoencoder Anomaly Detection Results")
    print(f"{'='*55}")
    print(f"  AUROC                : {auroc:.4f}")
    print(f"  AUPRC                : {auprc:.4f}")
    print(f"  Optimal threshold    : {best_thresh:.6f}")
    print(f"  F1  @ optimal thresh : {best_f1:.4f}")
    print(f"  Prec@ optimal thresh : {best_prec:.4f}")
    print(f"  Rec @ optimal thresh : {best_rec:.4f}")
    print(f"  Acc @ optimal thresh : {best_acc:.4f}")
    print(f"{'='*55}")
    print(f"  MSE normal   μ={metrics['reconstruction_mse_stats']['normal_mean']:.6f}  "
          f"σ={metrics['reconstruction_mse_stats']['normal_std']:.6f}")
    print(f"  MSE anomalous μ={metrics['reconstruction_mse_stats']['anomalous_mean']:.6f}  "
          f"σ={metrics['reconstruction_mse_stats']['anomalous_std']:.6f}")
    print(f"{'='*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN Autoencoder: train on normal images, evaluate anomaly detection.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train the autoencoder on normal images.")
    p_train.add_argument("--train-dir",   "-t", required=True,
                         help="Directory of normal training images (autoencoder_train/)")
    p_train.add_argument("--output-dir",  "-o", default="./autoencoder_output",
                         help="Where to save model + logs (default: ./autoencoder_output)")
    p_train.add_argument("--epochs",      type=int,   default=EPOCHS)
    p_train.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    p_train.add_argument("--lr",          type=float, default=LR)
    p_train.add_argument("--num-workers", type=int,   default=NUM_WORKERS)

    # ── eval ───────────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("eval", help="Evaluate a trained model on Data/images/.")
    p_eval.add_argument("--images-dir",   "-i", required=True,
                        help="Directory of evaluation images (Data/images/)")
    p_eval.add_argument("--dataset-json", "-d", required=True,
                        help="Path to Data/dataset.json")
    p_eval.add_argument("--model-path",   "-m", required=True,
                        help="Path to autoencoder.pth")
    p_eval.add_argument("--output-dir",   "-o", default="./autoencoder_output",
                        help="Where to write outputs (default: ./autoencoder_output)")
    p_eval.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    p_eval.add_argument("--num-workers",  type=int, default=NUM_WORKERS)

    # ── train_eval ─────────────────────────────────────────────────────────
    p_both = subparsers.add_parser("train_eval", help="Train then immediately evaluate.")
    p_both.add_argument("--train-dir",    "-t", required=True)
    p_both.add_argument("--images-dir",   "-i", required=True)
    p_both.add_argument("--dataset-json", "-d", required=True)
    p_both.add_argument("--output-dir",   "-o", default="./autoencoder_output")
    p_both.add_argument("--epochs",       type=int,   default=EPOCHS)
    p_both.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    p_both.add_argument("--lr",           type=float, default=LR)
    p_both.add_argument("--num-workers",  type=int,   default=NUM_WORKERS)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        train(
            train_dir=Path(args.train_dir),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
            num_workers=args.num_workers,
        )

    elif args.command == "eval":
        evaluate(
            images_dir=Path(args.images_dir),
            dataset_json=Path(args.dataset_json),
            model_path=Path(args.model_path),
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    elif args.command == "train_eval":
        model_path = train(
            train_dir=Path(args.train_dir),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
            num_workers=args.num_workers,
        )
        evaluate(
            images_dir=Path(args.images_dir),
            dataset_json=Path(args.dataset_json),
            model_path=model_path,
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
