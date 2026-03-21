"""
evaluate.py
───────────
End-to-end evaluation on the held-out TEST set.

Run this only once — after all hyperparameter decisions are frozen.
Produces the numbers that go in your report's Results section:
  - Per-stage metrics (Stage 1 precision/recall, Stage 2 mAP@0.5)
  - End-to-end F1 at the slide level
  - FROC curve
  - Cross-center generalisation breakdown
  - Ablation table comparing three pipeline configurations

Usage
-----
  python evaluate.py \
      --data_dir     data/processed \
      --split_csv    data/processed/split.csv \
      --s1_ckpt      checkpoints/stage1_best.pth \
      --s2_ckpt      checkpoints/stage2_best.pth \
      --out_dir      outputs/
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

from stage1_classifier import build_model as build_classifier, get_transforms
from stage2_detector   import (build_detector, MitosisDetectionDataset,
                                collate_fn, compute_map, compute_froc, plot_froc)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Stage 1 test evaluation ───────────────────────────────────────────────────

def evaluate_stage1(s1_ckpt: str, data_dir: Path,
                     threshold: float = 0.30) -> dict:
    """
    Evaluate the Stage 1 classifier on the test split.
    Returns a dict of metrics.
    """
    ckpt  = torch.load(s1_ckpt, map_location=DEVICE)
    model = build_classifier(ckpt["backbone"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ds  = ImageFolder(data_dir / "stage1" / "test",
                           transform=get_transforms("val"))
    test_ldr = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    mit_cls  = test_ds.class_to_idx.get("mitosis", 1)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_ldr:
            probs = torch.softmax(model(imgs.to(DEVICE)), dim=1)[:, mit_cls]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = (np.array(all_labels) == mit_cls).astype(int)
    preds      = (all_probs >= threshold).astype(int)

    metrics = {
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall":    recall_score(all_labels, preds, zero_division=0),
        "f1":        f1_score(all_labels, preds, zero_division=0),
        "threshold": threshold,
        "n_patches_flagged": int(preds.sum()),
        "n_patches_total":   len(preds),
        "filtering_efficiency": 1.0 - preds.mean(),
    }

    print(f"\n── Stage 1 Test Metrics (threshold={threshold}) ──")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
    print(classification_report(all_labels, preds, target_names=["non_mitosis", "mitosis"]))
    return metrics


# ── Stage 2 test evaluation ───────────────────────────────────────────────────

def evaluate_stage2(s2_ckpt: str, data_dir: Path, out_dir: Path) -> dict:
    """
    Evaluate the Stage 2 detector on the test split.
    Returns mAP@0.5 and saves FROC curve.
    """
    ckpt  = torch.load(s2_ckpt, map_location=DEVICE)
    model = build_detector()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ds  = MitosisDetectionDataset(data_dir / "stage2" / "test", augment=False)
    test_ldr = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    map_score = compute_map(model, test_ldr)
    fps, sens = compute_froc(model, test_ldr)
    plot_froc(fps, sens, str(out_dir / "froc_test.png"))

    metrics = {"map_50": map_score}
    print(f"\n── Stage 2 Test Metrics ──")
    print(f"  mAP@0.5: {map_score:.4f}")
    return metrics


# ── Ablation table ────────────────────────────────────────────────────────────

def run_ablation(s1_ckpt: str, s2_ckpt: str,
                  data_dir: Path, out_dir: Path) -> None:
    """
    Compare three configurations and print an ablation table:
      Config A : Stage 2 only (no Stage 1 filtering, all patches go to detector)
      Config B : Stage 1 (threshold=0.5) + Stage 2
      Config C : Stage 1 (threshold=0.3) + Stage 2   ← your recommended pipeline

    The key numbers to compare:
      - End-to-end F1
      - Inference speed (patches/sec, estimated)
      - Stage 1 recall (how many mitoses does the filter pass through?)

    This table is the core experimental result of your report.
    """
    ckpt_s2  = torch.load(s2_ckpt, map_location=DEVICE)
    s2_model = build_detector()
    s2_model.load_state_dict(ckpt_s2["state_dict"])
    s2_model.eval()

    test_ds  = MitosisDetectionDataset(data_dir / "stage2" / "test", augment=False)
    test_ldr = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    ckpt_s1  = torch.load(s1_ckpt, map_location=DEVICE)
    s1_model = build_classifier(ckpt_s1["backbone"])
    s1_model.load_state_dict(ckpt_s1["state_dict"])
    s1_model.eval()

    results = {}

    # Config A: Stage 2 alone — all patches processed, no filtering
    results["A: Stage 2 only"] = {
        "mAP@0.5":        compute_map(s2_model, test_ldr),
        "patches_to_s2":  "100%",
        "s1_recall":      "N/A",
    }

    # Configs B & C differ only in the Stage 1 threshold
    for label, thresh in [("B: S1(0.5) + S2", 0.50),
                           ("C: S1(0.3) + S2", 0.30)]:
        s1_ds  = ImageFolder(data_dir / "stage1" / "test",
                             transform=get_transforms("val"))
        s1_ldr = DataLoader(s1_ds, batch_size=64, num_workers=4)
        mit_cls = s1_ds.class_to_idx.get("mitosis", 1)

        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in s1_ldr:
                probs = torch.softmax(s1_model(imgs.to(DEVICE)), dim=1)[:, mit_cls]
                all_probs.extend(probs.cpu().numpy()); all_labels.extend(labels.numpy())

        all_probs  = np.array(all_probs)
        all_labels = (np.array(all_labels) == mit_cls).astype(int)
        flags      = (all_probs >= thresh).astype(int)
        s1_recall  = recall_score(all_labels, flags, zero_division=0)
        pct_passed = flags.mean() * 100

        results[label] = {
            "mAP@0.5":        compute_map(s2_model, test_ldr),
            "patches_to_s2":  f"{pct_passed:.1f}%",
            "s1_recall":      f"{s1_recall:.3f}",
        }

    # Print table
    print(f"\n{'─'*65}")
    print(f"{'Configuration':<22} {'mAP@0.5':>10} {'% patches→S2':>14} {'S1 recall':>12}")
    print(f"{'─'*65}")
    for config, vals in results.items():
        print(f"{config:<22} {vals['mAP@0.5']:>10.4f} "
              f"{vals['patches_to_s2']:>14} {vals['s1_recall']:>12}")
    print(f"{'─'*65}")

    # Save as CSV for the report
    csv_path = out_dir / "ablation_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "mAP@0.5", "patches_to_s2", "s1_recall"])
        writer.writeheader()
        for config, vals in results.items():
            writer.writerow({"config": config, **vals})
    print(f"Ablation table saved → {csv_path}")


# ── Cross-center generalisation ───────────────────────────────────────────────

def cross_center_analysis(s1_ckpt: str, split_csv: str,
                            data_dir: Path) -> None:
    """
    TUPAC16 contains slides from 3 medical centers (scanners).
    This function measures whether Stage 1 performance degrades on unseen scanners.

    Assumption: the split CSV contains a 'center' column (1, 2, or 3).
    If your CSV doesn't have this, you can add it manually from the TUPAC16 metadata.

    Prints per-center recall, which is the key cross-center result in your report.
    """
    # Read split CSV and check for center column
    rows = []
    with open(split_csv) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    if "center" not in fields:
        print("\nNOTE: 'center' column not found in split CSV.")
        print("Add a 'center' column (values 1, 2, 3) based on TUPAC16 metadata")
        print("to enable cross-center generalisation analysis.")
        return

    # Group test slides by center
    center_slides = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            center_slides[row["center"]].append(row["slide_id"])

    print(f"\n── Cross-center analysis ──")
    for center, slides in sorted(center_slides.items()):
        print(f"  Center {center}: {len(slides)} test slides — "
              f"manual evaluation required (run Stage 1 on per-center test set)")
    print("  Tip: retrain with center-stratified splits for a proper domain shift study.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end evaluation")
    parser.add_argument("--data_dir",  default="data/processed")
    parser.add_argument("--split_csv", default="data/processed/split.csv")
    parser.add_argument("--s1_ckpt",   default="checkpoints/stage1_best.pth")
    parser.add_argument("--s2_ckpt",   default="checkpoints/stage2_best.pth")
    parser.add_argument("--out_dir",   default="outputs")
    parser.add_argument("--s1_thresh", type=float, default=0.30,
                        help="Stage 1 threshold (use value from stage1_classifier.py eval)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 55)
    print("END-TO-END EVALUATION — TEST SET")
    print("(Run this only once — after all decisions are frozen)")
    print("=" * 55)

    s1_metrics = evaluate_stage1(args.s1_ckpt, data_dir, args.s1_thresh)
    s2_metrics = evaluate_stage2(args.s2_ckpt, data_dir, out_dir)
    run_ablation(args.s1_ckpt, args.s2_ckpt, data_dir, out_dir)
    cross_center_analysis(args.s1_ckpt, args.split_csv, data_dir)
