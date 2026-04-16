"""
stage1_classifier.py
Trains and evaluates the Stage 1 binary classifier:
  input  : 64x64 RGB patch
  output : probability that the patch contains a mitotic figure

Architecture choices
- ResNet50 pretrained on ImageNet.
- Final FC layer replaced with a 2-class head.
- Training at high RECALL is the explicit goal: a missed mitosis (FN) is worse
  than a false alarm (FP) because Stage 2 will filter out the FPs anyway.

Usage
  # Train
  python stage1_classifier.py --mode train \
      --data_dir data/processed/stage1 \
      --backbone resnet50 \
      --epochs 30 --batch_size 64 --lr 1e-4

  # Evaluate on val set and pick best threshold
  python stage1_classifier.py --mode eval \
      --data_dir data/processed/stage1 \
      --checkpoint checkpoints/stage1_best.pth
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from sklearn.metrics import (classification_report, precision_recall_curve,
                              roc_auc_score, f1_score)
import matplotlib.pyplot as plt


DEFAULT_PATCH_SIZE = 64
DEFAULT_LR         = 1e-4
DEFAULT_EPOCHS     = 30
DEFAULT_BATCH      = 64
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss  = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt       = torch.exp(-ce_loss)
        alpha_t  = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal    = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


def build_model(backbone: str = "resnet50", freeze_backbone: bool = False) -> nn.Module:
    """Build a binary classifier on top of a pretrained backbone."""
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 2)
        )
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    elif backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 2)
        )
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'resnet50' or 'efficientnet_b3'.")

    return model.to(DEVICE)


def get_transforms(split: str) -> T.Compose:
    """Get augmentation transforms for train/val splits."""
    if split == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ElasticTransform(alpha=30.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


def make_weighted_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    """Create weighted sampler for class balance."""
    class_counts  = np.bincount(dataset.targets)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = [class_weights[t] for t in dataset.targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train(args: argparse.Namespace) -> None:
    data_dir    = Path(args.data_dir)
    ckpt_dir    = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ImageFolder(data_dir / "train", transform=get_transforms("train"))
    val_ds   = ImageFolder(data_dir / "val",   transform=get_transforms("val"))

    print(f"Class → index mapping: {train_ds.class_to_idx}")
    print(f"  Train: {len(train_ds)} patches  |  Val: {len(val_ds)} patches")
    print(f"  Train class counts: {np.bincount(train_ds.targets)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_weighted_sampler(train_ds),
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model     = build_model(args.backbone, freeze_backbone=args.freeze_backbone)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    history     = {"train_loss": [], "val_f1": [], "val_recall": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(train_ds)

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                probs = torch.softmax(model(imgs), dim=1)[:, 1]  # P(mitosis)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)

        preds   = (all_probs >= 0.3).astype(int)
        val_f1  = f1_score(all_labels, preds, pos_label=1, zero_division=0)
        val_rec = (preds[all_labels == 1] == 1).mean() if all_labels.sum() > 0 else 0.0

        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_rec)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  val_F1={val_f1:.4f}  val_recall={val_rec:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best checkpoint by F1 (not just accuracy)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Use different checkpoint names for different backbones
            if args.backbone == "efficientnet_b3":
                ckpt_filename = "stage1_best_efficientnet_b3.pth"
            else:
                ckpt_filename = "stage1_best.pth"
            ckpt_path = ckpt_dir / ckpt_filename
            torch.save({
                "epoch":      epoch,
                "backbone":   args.backbone,
                "state_dict": model.state_dict(),
                "val_f1":     val_f1,
                "val_recall": val_rec,
            }, ckpt_path)
            print(f"  ✓ New best checkpoint saved → {ckpt_path}")

    # Save training curves
    _plot_training_curves(history, ckpt_dir / "stage1_curves.png")
    print(f"\nTraining complete. Best val F1 = {best_val_f1:.4f}")


def evaluate(args: argparse.Namespace) -> float:
    """Evaluate model on validation set and find optimal threshold."""
    data_dir = Path(args.data_dir)
    ckpt     = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

    model = build_model(ckpt["backbone"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_F1={ckpt['val_f1']:.4f})")

    val_ds     = ImageFolder(data_dir / "val", transform=get_transforms("val"))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    mitosis_class = val_ds.class_to_idx.get("mitosis", 1)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            probs = torch.softmax(model(imgs.to(DEVICE)), dim=1)[:, mitosis_class]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    all_labels = (all_labels == mitosis_class).astype(int)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nAUC-ROC: {auc:.4f}")

    high_recall_mask = recall[:-1] >= 0.90
    if high_recall_mask.any():
        best_idx   = np.argmax(precision[:-1][high_recall_mask])
        best_thresh = thresholds[high_recall_mask][best_idx]
        best_prec   = precision[:-1][high_recall_mask][best_idx]
        best_rec    = recall[:-1][high_recall_mask][best_idx]
    else:
        # Fallback: maximise F1
        f1_scores   = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
        best_idx    = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_prec   = precision[best_idx]
        best_rec    = recall[best_idx]

    print(f"Recommended threshold: {best_thresh:.3f}")
    print(f"  Precision @ threshold: {best_prec:.4f}")
    print(f"  Recall    @ threshold: {best_rec:.4f}")
    print(f"\n  Hard-code this threshold into pipeline.py as STAGE1_THRESHOLD\n")

    preds = (all_probs >= best_thresh).astype(int)
    print(classification_report(all_labels, preds, target_names=["non_mitosis", "mitosis"]))

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="#534AB7", linewidth=2, label=f"AUC={auc:.3f}")
    plt.axvline(best_rec, color="#D85A30", linestyle="--", alpha=0.7,
                label=f"threshold={best_thresh:.2f}  recall={best_rec:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Stage 1 — Precision-Recall Curve")
    plt.legend(); plt.tight_layout()
    out_path = Path(args.checkpoint).parent / "stage1_pr_curve.png"
    plt.savefig(out_path, dpi=150)
    print(f"PR curve saved → {out_path}")

    return best_thresh


def gradcam_visualise(model: nn.Module,
                       img_tensor: torch.Tensor,
                       target_class: int = 1) -> np.ndarray:
    """Produce Grad-CAM heatmap for visualization."""
    model.eval()
    activations, gradients = [], []

    def _fwd_hook(module, input, output):
        activations.append(output.detach())

    def _bwd_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    if hasattr(model, "layer4"):
        hook_layer = model.layer4[-1]
    else:
        hook_layer = model.features[-1]

    fh = hook_layer.register_forward_hook(_fwd_hook)
    bh = hook_layer.register_full_backward_hook(_bwd_hook)

    img_tensor = img_tensor.to(DEVICE)
    logits = model(img_tensor)
    model.zero_grad()
    logits[0, target_class].backward()

    fh.remove(); bh.remove()

    act  = activations[0].squeeze()
    grad = gradients[0].squeeze()
    weights = grad.mean(dim=(1, 2))

    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = torch.relu(cam).cpu().numpy()
    cam = cam - cam.min()
    denom = cam.max()
    if denom > 0:
        cam /= denom
    return cam


def _plot_training_curves(history: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], color="#534AB7")
    axes[0].set_title("Training loss"); axes[0].set_xlabel("Epoch")
    axes[1].plot(history["val_f1"],    color="#1D9E75", label="F1")
    axes[1].plot(history["val_recall"],color="#D85A30", label="Recall", linestyle="--")
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 — patch classifier")
    parser.add_argument("--mode",           choices=["train", "eval"], required=True)
    parser.add_argument("--data_dir",       default="data/processed/stage1")
    parser.add_argument("--backbone",       default="resnet50",
                        choices=["resnet50", "efficientnet_b3"])
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint",     default="checkpoints/stage1_best.pth",
                        help="Path to .pth file (eval mode only)")
    parser.add_argument("--epochs",         type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",     type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",             type=float, default=DEFAULT_LR)
    parser.add_argument("--freeze_backbone",action="store_true",
                        help="Freeze backbone weights for first training run")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
