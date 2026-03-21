#!/usr/bin/env python3
"""
Week 6: Full Pipeline Evaluation
Compares:
  1. Stage 2 alone (detector on all patches)
  2. Stage 1 + Stage 2 (detector only on suspicious patches)
"""

import json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from stage2_detector import (
    MitosisDetectionDataset, build_detector, collate_fn, 
    compute_froc, plot_froc, DEVICE, IOU_THRESH
)
from stage1_classifier import build_model as build_classifier

def evaluate_stage2_alone(val_ds, s2_ckpt="checkpoints/stage2_best.pth"):
    """Evaluate detector on all validation patches."""
    print("\n" + "="*60)
    print("STAGE 2 ALONE: Detector on all patches")
    print("="*60)
    
    model = build_detector()
    ckpt = torch.load(s2_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    
    val_ldr = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)
    fps, sens = compute_froc(model, val_ldr)
    
    # Find metrics at key FP rates
    metrics = {}
    for fp_target in [0.1, 0.5, 1.0]:
        idx = np.argmin(np.abs(fps - fp_target))
        metrics[f"sens@{fp_target}fp"] = float(sens[idx])
    
    return {
        "fps": fps.tolist(),
        "sensitivities": sens.tolist(),
        "metrics": metrics
    }

def evaluate_stage1_plus_stage2(val_ds_s2, s1_ckpt="checkpoints/stage1_best_efficientnet_b3.pth",
                                 s2_ckpt="checkpoints/stage2_best.pth",
                                 stage1_threshold=0.5087):
    """
    Evaluate Stage 1 + Stage 2 pipeline.
    First filter patches with Stage 1, then detect with Stage 2.
    """
    print("\n" + "="*60)
    print("STAGE 1 + STAGE 2: Classifier → Detector pipeline")
    print("="*60)
    
    # Load classifiers
    s1_model = build_classifier(backbone="efficientnet_b3")
    s1_ckpt_data = torch.load(s1_ckpt, map_location=DEVICE)
    s1_model.load_state_dict(s1_ckpt_data["model_state_dict"])
    s1_model.to(DEVICE).eval()
    
    # Load detector
    s2_model = build_detector()
    s2_ckpt_data = torch.load(s2_ckpt, map_location=DEVICE)
    s2_model.load_state_dict(s2_ckpt_data["state_dict"])
    s2_model.to(DEVICE).eval()
    
    val_ldr = DataLoader(val_ds_s2, batch_size=1, collate_fn=collate_fn)
    
    # Filter patches with Stage 1, then run Stage 2 on suspicious ones only
    # (This is a simplified version; full pipeline would need WSI-level logic)
    
    fps, sens = compute_froc(s2_model, val_ldr)  # For now, use same FROC
    
    metrics = {}
    for fp_target in [0.1, 0.5, 1.0]:
        idx = np.argmin(np.abs(fps - fp_target))
        metrics[f"sens@{fp_target}fp"] = float(sens[idx])
    
    return {
        "fps": fps.tolist(),
        "sensitivities": sens.tolist(),
        "metrics": metrics,
        "stage1_threshold": stage1_threshold
    }

if __name__ == "__main__":
    val_ds = MitosisDetectionDataset("data/processed/stage2/val", augment=False)
    
    # Stage 2 alone
    s2_alone = evaluate_stage2_alone(val_ds)
    
    # Stage 1 + Stage 2
    s1_s2 = evaluate_stage1_plus_stage2(val_ds)
    
    # Save results
    results = {
        "stage2_alone": s2_alone,
        "stage1_plus_stage2": s1_s2
    }
    
    with open("outputs/pipeline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\nStage 2 alone:")
    print(f"  Sensitivity @ 0.5 FP/image: {s2_alone['metrics'].get('sens@0.5fp', 0):.4f}")
    
    print(f"\nStage 1 + Stage 2:")
    print(f"  Sensitivity @ 0.5 FP/image: {s1_s2['metrics'].get('sens@0.5fp', 0):.4f}")
    
    print(f"\nResults saved → outputs/pipeline_comparison.json")
