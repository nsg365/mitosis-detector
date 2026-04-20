import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.append("src")
from stage1_classifier import build_model, get_transforms, DEVICE

def main():
    data_dir = Path("data/processed/stage1")
    ckpt = torch.load("checkpoints/stage1_best.pth", map_location=DEVICE, weights_only=False)
    
    model = build_model(ckpt["backbone"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    val_ds = ImageFolder(data_dir / "val", transform=get_transforms("val"))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    
    mitosis_class = val_ds.class_to_idx.get("mitosis", 1)
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            probs = torch.softmax(model(imgs.to(DEVICE)), dim=1)[:, mitosis_class]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_labels = (np.array(all_labels) == mitosis_class).astype(int)
    
    thresholds = np.linspace(0.05, 0.95, 50)
    f1_scores = []
    filter_rates = []
    
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        filter_rate = (preds == 0).mean() * 100
        f1_scores.append(f1)
        filter_rates.append(filter_rate)
        
    best_t = 0.5087 # the chosen threshold
    best_preds = (all_probs >= best_t).astype(int)
    best_f1 = f1_score(all_labels, best_preds, zero_division=0)
    best_filter = (best_preds == 0).mean() * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: F1 Score
    axes[0].plot(thresholds, f1_scores, color="#534AB7", linewidth=2)
    axes[0].axvline(best_t, color="#D85A30", linestyle="--", label=f"Selected $\\tau={best_t}$")
    axes[0].scatter([best_t], [best_f1], color="red", zorder=5)
    axes[0].set_xlabel("Stage 1 Threshold")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("F1 Score vs Stage 1 Threshold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Filtering Rate
    axes[1].plot(thresholds, filter_rates, color="#1D9E75", linewidth=2)
    axes[1].axvline(best_t, color="#D85A30", linestyle="--", label=f"Selected $\\tau={best_t}$")
    axes[1].scatter([best_t], [best_filter], color="red", zorder=5)
    axes[1].set_xlabel("Stage 1 Threshold")
    axes[1].set_ylabel("Patches Filtered Out (%)")
    axes[1].set_title("Filtering Rate vs Threshold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    out_path = Path("outputs/stage1_threshold_analysis.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print("Graph saved to", out_path)

if __name__ == "__main__":
    main()
