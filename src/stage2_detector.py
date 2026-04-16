"""
stage2_detector.py
Trains and evaluates the Stage 2 object detector:
  input  : 512x512 RGB patch flagged as suspicious by Stage 1
  output : bounding boxes around individual mitotic figures

Architecture
torchvision Faster R-CNN with a ResNet50-FPN backbone pretrained on COCO.
We replace the box predictor head with a 2-class version (background + mitosis)
and tune the anchor generator for the small size of mitotic figures (~20–30px).

Label format (from preprocess.py)
Each .txt label file has one line per box:
    x_min  y_min  x_max  y_max  class_id
All in absolute pixel coordinates (not normalised). class_id = 1 for mitosis.

Usage
  # Train
  python stage2_detector.py --mode train \
      --data_dir data/processed/stage2 \
      --epochs 100 --batch_size 4 --lr 5e-4

  # Evaluate on val patches
  python stage2_detector.py --mode eval \
      --data_dir data/processed/stage2 \
      --checkpoint checkpoints/stage2_best.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torchvision.models.detection as det_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
IOU_THRESH  = 0.5

ANCHOR_SIZES   = ((8,), (16,), (32,), (64,), (128,))
ANCHOR_RATIOS  = ((0.5, 1.0, 2.0),) * 5


class MitosisDetectionDataset(Dataset):

    def __init__(self, root: str, augment: bool = False):
        self.img_dir  = Path(root) / "images"
        self.lbl_dir  = Path(root) / "labels"
        self.augment  = augment
        self.stems    = sorted([p.stem for p in self.img_dir.glob("*.png")])

        if not self.stems:
            raise FileNotFoundError(f"No PNG images found in {self.img_dir}")
        print(f"  Dataset: {len(self.stems)} patches in {root}")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]

        img   = Image.open(self.img_dir / f"{stem}.png").convert("RGB")
        img_t = TF.to_tensor(img)


        lbl_path = self.lbl_dir / f"{stem}.txt"
        boxes, labels = [], []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    x1, y1, x2, y2 = map(float, parts[:4])
                    cls = int(parts[4])
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":     boxes_t,
            "labels":    labels_t,
            "image_id":  torch.tensor([idx]),
        }

        if self.augment:
            if torch.rand(1) > 0.5:
                img_t, target = _hflip(img_t, target)
            if torch.rand(1) > 0.5:
                img_t, target = _vflip(img_t, target)

        return img_t, target


def _hflip(img: torch.Tensor, target: dict) -> tuple:
    """Horizontally flip image and adjust bounding boxes."""
    img = TF.hflip(img)
    _, h, w = img.shape
    boxes = target["boxes"].clone()
    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    target["boxes"] = boxes
    return img, target


def _vflip(img: torch.Tensor, target: dict) -> tuple:
    """Vertically flip image and adjust bounding boxes."""
    img = TF.vflip(img)
    _, h, w = img.shape
    boxes = target["boxes"].clone()
    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    target["boxes"] = boxes
    return img, target


def collate_fn(batch):
    """
    Faster R-CNN expects a list of (image, target) tuples — not stacked tensors —
    because each image may have a different number of boxes.
    """
    return tuple(zip(*batch))


def build_detector() -> torch.nn.Module:
    anchor_generator = AnchorGenerator(
        sizes=ANCHOR_SIZES,
        aspect_ratios=ANCHOR_RATIOS,
    )

    model = det_models.fasterrcnn_resnet50_fpn(
        weights=det_models.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        rpn_anchor_generator=anchor_generator,
        rpn_fg_iou_thresh=0.5,
        rpn_bg_iou_thresh=0.3,
        rpn_nms_thresh=0.6,
        box_nms_thresh=0.3,
        box_score_thresh=0.3,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    return model.to(DEVICE)


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MitosisDetectionDataset(data_dir / "train", augment=True)
    val_ds   = MitosisDetectionDataset(data_dir / "val",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    model     = build_detector()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_map = 0.0
    history      = {"train_loss": [], "val_map": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, targets in train_loader:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        val_map = compute_map(model, val_loader)
        history["train_loss"].append(avg_loss)
        history["val_map"].append(val_map)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  val_mAP@0.5={val_map:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_map > best_val_map:
            best_val_map = val_map
            ckpt_path    = ckpt_dir / "stage2_best.pth"
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_map":    val_map,
            }, ckpt_path)
            print(f"  ✓ New best checkpoint saved → {ckpt_path}")

    _plot_detection_curves(history, ckpt_dir / "stage2_curves.png")
    print(f"\nTraining complete. Best val mAP@0.5 = {best_val_map:.4f}")


def compute_map(model: torch.nn.Module,
                loader: DataLoader,
                iou_threshold: float = IOU_THRESH) -> float:
    model.eval()
    all_scores, all_tp, all_fp = [], [], []
    n_gt_total = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(DEVICE) for img in imgs]
            preds = model(imgs)

            for pred, target in zip(preds, targets):
                gt_boxes   = target["boxes"].numpy()
                pred_boxes = pred["boxes"].cpu().numpy()
                scores     = pred["scores"].cpu().numpy()
                pred_lbls  = pred["labels"].cpu().numpy()

                mit_mask   = pred_lbls == 1
                pred_boxes = pred_boxes[mit_mask]
                scores     = scores[mit_mask]

                n_gt = len(gt_boxes)
                n_gt_total += n_gt
                matched_gt = set()

                sort_idx = np.argsort(-scores)
                for i in sort_idx:
                    if n_gt == 0:
                        all_tp.append(0); all_fp.append(1)
                        all_scores.append(scores[i])
                        continue
                    ious = _iou_batch(pred_boxes[i], gt_boxes)
                    best_iou_idx = int(np.argmax(ious))
                    best_iou     = ious[best_iou_idx]

                    if best_iou >= iou_threshold and best_iou_idx not in matched_gt:
                        all_tp.append(1); all_fp.append(0)
                        matched_gt.add(best_iou_idx)
                    else:
                        all_tp.append(0); all_fp.append(1)
                    all_scores.append(scores[i])

    if not all_scores or n_gt_total == 0:
        return 0.0

    order      = np.argsort(-np.array(all_scores))
    cum_tp     = np.cumsum(np.array(all_tp)[order])
    cum_fp     = np.cumsum(np.array(all_fp)[order])
    precision  = cum_tp / (cum_tp + cum_fp + 1e-9)
    recall     = cum_tp / (n_gt_total + 1e-9)

    ap = 0.0
    for r_thresh in np.linspace(0, 1, 11):
        mask = recall >= r_thresh
        ap  += precision[mask].max() if mask.any() else 0.0
    return ap / 11.0


def _iou_batch(box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a single predicted box and all GT boxes."""
    x1 = np.maximum(box[0], gt_boxes[:, 0])
    y1 = np.maximum(box[1], gt_boxes[:, 1])
    x2 = np.minimum(box[2], gt_boxes[:, 2])
    y2 = np.minimum(box[3], gt_boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt   = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    return inter / (area_pred + area_gt - inter + 1e-9)


def compute_froc(model: torch.nn.Module,
                 loader: DataLoader,
                 score_thresholds: np.ndarray = None) -> tuple:
    if score_thresholds is None:
        score_thresholds = np.linspace(0.05, 0.95, 40)

    model.eval()
    all_preds_by_thresh = {t: {"tp": 0, "fp": 0} for t in score_thresholds}
    n_images = 0
    n_gt_total = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs   = [img.to(DEVICE) for img in imgs]
            preds  = model(imgs)
            n_images += len(imgs)

            for pred, target in zip(preds, targets):
                gt_boxes   = target["boxes"].numpy()
                pred_boxes = pred["boxes"].cpu().numpy()
                scores     = pred["scores"].cpu().numpy()
                pred_lbls  = pred["labels"].cpu().numpy()

                mit_mask   = pred_lbls == 1
                pred_boxes = pred_boxes[mit_mask]
                scores     = scores[mit_mask]
                n_gt_total += len(gt_boxes)

                for thresh in score_thresholds:
                    keep       = scores >= thresh
                    kept_boxes = pred_boxes[keep]
                    matched_gt = set()
                    tp = fp = 0

                    for box in kept_boxes:
                        if len(gt_boxes) == 0:
                            fp += 1; continue
                        ious         = _iou_batch(box, gt_boxes)
                        best_idx     = int(np.argmax(ious))
                        if ious[best_idx] >= IOU_THRESH and best_idx not in matched_gt:
                            tp += 1; matched_gt.add(best_idx)
                        else:
                            fp += 1

                    all_preds_by_thresh[thresh]["tp"] += tp
                    all_preds_by_thresh[thresh]["fp"] += fp

    fps_per_image, sensitivities = [], []
    for thresh in score_thresholds:
        tp = all_preds_by_thresh[thresh]["tp"]
        fp = all_preds_by_thresh[thresh]["fp"]
        fps_per_image.append(fp / max(n_images, 1))
        sensitivities.append(tp / max(n_gt_total, 1))

    return np.array(fps_per_image), np.array(sensitivities)


def plot_froc(fps: np.ndarray, sens: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(fps, sens, color="#534AB7", linewidth=2, marker="o", markersize=4)
    plt.xlabel("Average false positives per image")
    plt.ylabel("Sensitivity (recall)")
    plt.title("FROC curve — Stage 2 detector")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"FROC curve saved → {save_path}")


def visualise_predictions(model: torch.nn.Module,
                            dataset: MitosisDetectionDataset,
                            n_samples: int = 6,
                            score_thresh: float = 0.4,
                            save_path: str = "outputs/stage2_predictions.png") -> None:
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            img_t, target = dataset[idx]
            pred  = model([img_t.to(DEVICE)])[0]

            img_np = img_t.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)

            for box in target["boxes"].numpy():
                rect = mpatches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=1.5, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)

            for box, score, lbl in zip(
                    pred["boxes"].cpu().numpy(),
                    pred["scores"].cpu().numpy(),
                    pred["labels"].cpu().numpy()):
                if score >= score_thresh and lbl == 1:
                    rect = mpatches.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=1.5, edgecolor="red", facecolor="none")
                    ax.add_patch(rect)
                    ax.text(box[0], box[1] - 2, f"{score:.2f}",
                            color="red", fontsize=7)

            ax.axis("off")

    gt_patch   = mpatches.Patch(color="lime", label="Ground truth")
    pred_patch = mpatches.Patch(color="red",  label="Prediction")
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center",
               ncol=2, fontsize=10)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Prediction visualisation saved → {save_path}")


def _plot_detection_curves(history: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], color="#534AB7")
    axes[0].set_title("Training loss"); axes[0].set_xlabel("Epoch")
    axes[1].plot(history["val_map"], color="#1D9E75")
    axes[1].set_title("Val mAP@0.5"); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 — Faster R-CNN detector")
    parser.add_argument("--mode",           choices=["train", "eval"], required=True)
    parser.add_argument("--data_dir",       default="data/processed/stage2")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint",     default="checkpoints/stage2_best.pth")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=5e-4)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        model = build_detector()
        model.load_state_dict(ckpt["state_dict"])

        data_dir = Path(args.data_dir)
        val_ds   = MitosisDetectionDataset(data_dir / "val", augment=False)
        val_ldr  = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

        fps, sens = compute_froc(model, val_ldr)
        plot_froc(fps, sens, "outputs/froc_curve.png")
        visualise_predictions(model, val_ds)
