"""
pipeline.py
───────────
End-to-end inference: WSI → Stage 1 filter → Stage 2 detect → annotated output.

This is the file that wires the two trained models together.
It also produces the slide-level mitosis count, which maps directly to the
Nottingham Grade mitosis score (1/2/3) used in clinical reporting.

Usage
-----
  python pipeline.py \
      --wsi_path  data/raw/wsis/slide_001.tif \
      --s1_ckpt   checkpoints/stage1_best.pth \
      --s2_ckpt   checkpoints/stage2_best.pth \
      --out_dir   outputs/

Key hyperparameter to set
--------------------------
STAGE1_THRESHOLD : Set this to the value printed by stage1_classifier.py --mode eval.
                   Default here is 0.3 (recall-optimised). Adjust after evaluation.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import openslide
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from preprocess import (macenko_normalise, get_level_for_magnification,
                         read_patch, PATCH_SIZE_S1, PATCH_SIZE_S2, STRIDE_S1)
from stage1_classifier import build_model as build_classifier
from stage2_detector import build_detector


# ── IMPORTANT: set this after running stage1_classifier.py --mode eval ────────
STAGE1_THRESHOLD = 0.5087   # EfficientNet-B3: P(mitosis) threshold for Stage 2
# Achieved 100% recall (0% FNR) on validation set
# ─────────────────────────────────────────────────────────────────────────────

STAGE2_SCORE_THRESH = 0.40  # Faster R-CNN confidence threshold for final detections
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Transform for Stage 1 inference ───────────────────────────────────────────

S1_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_stage1(ckpt_path: str) -> torch.nn.Module:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = build_classifier(ckpt["backbone"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Stage 1 loaded  (backbone={ckpt['backbone']}, "
          f"val_F1={ckpt.get('val_f1', '?'):.4f})")
    return model


def load_stage2(ckpt_path: str) -> torch.nn.Module:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = build_detector()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Stage 2 loaded  (val_mAP={ckpt.get('val_map', '?'):.4f})")
    return model


# ── Sliding window patch generator ────────────────────────────────────────────

def sliding_window_patches(slide: openslide.OpenSlide,
                             level: int,
                             patch_size: int,
                             stride: int):
    """
    Generator that yields (patch_rgb, x0, y0) tuples for every stride step
    across the WSI at level-0 coordinates (x0, y0).
    Patches that are >80% background are skipped inside read_patch().
    """
    w0, h0 = slide.level_dimensions[0]
    for y0 in range(0, h0 - patch_size + 1, stride):
        for x0 in range(0, w0 - patch_size + 1, stride):
            patch = read_patch(slide, x0, y0, patch_size, level)
            if patch is not None:
                yield patch, x0, y0


# ── Stage 1: flag suspicious patches ──────────────────────────────────────────

@torch.no_grad()
def run_stage1(slide: openslide.OpenSlide,
                level: int,
                model: torch.nn.Module,
                batch_size: int = 64) -> list[tuple[np.ndarray, int, int]]:
    """
    Slide the 64×64 window over the WSI and collect patches whose
    P(mitosis) exceeds STAGE1_THRESHOLD.

    Returns a list of (patch_rgb_64, x0_level0, y0_level0).
    """
    suspicious = []
    batch_patches, batch_coords = [], []
    total_processed = total_flagged = 0

    for patch, x0, y0 in sliding_window_patches(
            slide, level, PATCH_SIZE_S1, STRIDE_S1):

        patch_norm = macenko_normalise(patch)
        img_pil    = Image.fromarray(patch_norm)
        tensor     = S1_TRANSFORM(img_pil)
        batch_patches.append(tensor)
        batch_coords.append((patch_norm, x0, y0))
        total_processed += 1

        if len(batch_patches) == batch_size:
            batch_t = torch.stack(batch_patches).to(DEVICE)
            probs   = torch.softmax(model(batch_t), dim=1)[:, 1]  # P(mitosis)
            for prob, coord in zip(probs.cpu().numpy(), batch_coords):
                if prob >= STAGE1_THRESHOLD:
                    suspicious.append(coord)
                    total_flagged += 1
            batch_patches, batch_coords = [], []

    # Process remaining patches
    if batch_patches:
        batch_t = torch.stack(batch_patches).to(DEVICE)
        probs   = torch.softmax(model(batch_t), dim=1)[:, 1]
        for prob, coord in zip(probs.cpu().numpy(), batch_coords):
            if prob >= STAGE1_THRESHOLD:
                suspicious.append(coord)
                total_flagged += 1
        total_processed += len(batch_patches)

    print(f"  Stage 1: {total_processed} patches processed, "
          f"{total_flagged} flagged ({100*total_flagged/max(total_processed,1):.1f}%)")
    return suspicious


# ── Stage 2: detect mitoses in flagged patches ─────────────────────────────────

@torch.no_grad()
def run_stage2(suspicious_patches: list[tuple[np.ndarray, int, int]],
                slide: openslide.OpenSlide,
                level: int,
                model: torch.nn.Module) -> list[dict]:
    """
    For each suspicious 64×64 patch, extract the corresponding 512×512 context
    patch (centred on the same location) and run Faster R-CNN.

    Returns a list of detection dicts:
        {
            "box_slide":   (x1, y1, x2, y2) in level-0 slide coordinates,
            "box_patch":   (x1, y1, x2, y2) in patch-local pixels (512×512),
            "score":       float confidence,
            "patch_rgb":   512×512 RGB ndarray,
            "patch_origin":(x0, y0) level-0 origin of the 512×512 patch,
        }
    """
    detections = []
    half_512   = PATCH_SIZE_S2 // 2
    w0, h0     = slide.level_dimensions[0]

    for _, cx64, cy64 in suspicious_patches:
        # Centre a 512×512 patch on the 64×64 patch location
        cx = cx64 + PATCH_SIZE_S1 // 2
        cy = cy64 + PATCH_SIZE_S1 // 2
        ox = int(np.clip(cx - half_512, 0, w0 - PATCH_SIZE_S2))
        oy = int(np.clip(cy - half_512, 0, h0 - PATCH_SIZE_S2))

        patch512 = read_patch(slide, ox, oy, PATCH_SIZE_S2, level)
        if patch512 is None:
            continue
        patch512 = macenko_normalise(patch512)

        img_t = TF.to_tensor(Image.fromarray(patch512)).to(DEVICE)
        preds = model([img_t])[0]

        for box, score, lbl in zip(
                preds["boxes"].cpu().numpy(),
                preds["scores"].cpu().numpy(),
                preds["labels"].cpu().numpy()):

            if lbl != 1 or score < STAGE2_SCORE_THRESH:
                continue

            # Convert patch-local coordinates to slide-level coordinates
            x1_slide = ox + int(box[0])
            y1_slide = oy + int(box[1])
            x2_slide = ox + int(box[2])
            y2_slide = oy + int(box[3])

            detections.append({
                "box_slide":    (x1_slide, y1_slide, x2_slide, y2_slide),
                "box_patch":    tuple(box.astype(int)),
                "score":        float(score),
                "patch_rgb":    patch512,
                "patch_origin": (ox, oy),
            })

    print(f"  Stage 2: {len(detections)} raw detections from "
          f"{len(suspicious_patches)} suspicious patches")
    return detections


# ── Slide-level NMS ───────────────────────────────────────────────────────────

def slide_level_nms(detections: list[dict],
                     iou_threshold: float = 0.3) -> list[dict]:
    """
    Apply Non-Maximum Suppression at the slide level to remove duplicate
    detections that occur when overlapping patches both flag the same mitosis.
    """
    if not detections:
        return []

    boxes  = np.array([d["box_slide"] for d in detections], dtype=float)
    scores = np.array([d["score"]     for d in detections], dtype=float)

    # Torchvision NMS
    boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)

    from torchvision.ops import nms as tv_nms
    keep = tv_nms(boxes_t, scores_t, iou_threshold).numpy()

    kept = [detections[i] for i in keep]
    print(f"  After slide-level NMS: {len(kept)} final detections "
          f"(removed {len(detections) - len(kept)} duplicates)")
    return kept


# ── Nottingham mitosis score ───────────────────────────────────────────────────

def nottingham_mitosis_score(n_mitoses: int) -> int:
    """
    Convert a mitosis count per 10 high-power fields (HPF) to a Nottingham score.
    TUPAC16 patches are at 40×, so one 512×512 patch ≈ 1 HPF.

    Scoring:
      1  ≤ 7 mitoses per 10 HPF
      2  8–14 mitoses per 10 HPF
      3  ≥ 15 mitoses per 10 HPF
    """
    if n_mitoses <= 7:
        return 1
    elif n_mitoses <= 14:
        return 2
    else:
        return 3


# ── Output: annotated thumbnail ────────────────────────────────────────────────

def save_annotated_thumbnail(slide: openslide.OpenSlide,
                              detections: list[dict],
                              save_path: str,
                              thumb_size: tuple = (2000, 2000)) -> None:
    """
    Generate an annotated slide thumbnail with all final detections overlaid.
    Red dots mark detected mitoses. This is the key qualitative figure for your report.
    """
    thumbnail = slide.get_thumbnail(thumb_size)
    thumb_np  = np.array(thumbnail.convert("RGB"))

    # Scale factor from level-0 to thumbnail
    w0, h0   = slide.level_dimensions[0]
    scale_x  = thumb_size[0] / w0
    scale_y  = thumb_size[1] / h0

    thumb_bgr = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1, x2, y2 = det["box_slide"]
        tx1 = int(x1 * scale_x); ty1 = int(y1 * scale_y)
        tx2 = int(x2 * scale_x); ty2 = int(y2 * scale_y)
        cx  = (tx1 + tx2) // 2;  cy  = (ty1 + ty2) // 2

        cv2.circle(thumb_bgr, (cx, cy), radius=6,
                   color=(0, 0, 220), thickness=-1)           # red dot
        cv2.circle(thumb_bgr, (cx, cy), radius=8,
                   color=(255, 255, 255), thickness=1)         # white ring

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, thumb_bgr)
    print(f"Annotated thumbnail saved → {save_path}")


# ── Main inference function ────────────────────────────────────────────────────

def run_pipeline(wsi_path: str,
                  s1_ckpt:  str,
                  s2_ckpt:  str,
                  out_dir:  str) -> dict:
    """
    Full pipeline inference on a single WSI.

    Returns a result dict with keys:
        n_mitoses, nottingham_score, detections, slide_id
    """
    out_dir  = Path(out_dir)
    slide_id = Path(wsi_path).stem

    print(f"\n{'═'*55}")
    print(f"Running pipeline on: {slide_id}")
    print(f"  Stage 1 threshold : {STAGE1_THRESHOLD}")
    print(f"  Stage 2 threshold : {STAGE2_SCORE_THRESH}")
    print(f"{'═'*55}")

    s1_model = load_stage1(s1_ckpt)
    s2_model = load_stage2(s2_ckpt)

    slide = openslide.open_slide(wsi_path)
    level = get_level_for_magnification(slide)
    print(f"  WSI level: {level}, dims: {slide.level_dimensions[level]}")

    # Stage 1
    suspicious = run_stage1(slide, level, s1_model)

    # Stage 2
    raw_detections = run_stage2(suspicious, slide, level, s2_model)

    # Slide-level NMS
    final_detections = slide_level_nms(raw_detections)

    n_mitoses = len(final_detections)
    score     = nottingham_mitosis_score(n_mitoses)

    print(f"\n{'─'*55}")
    print(f"RESULT — {slide_id}")
    print(f"  Mitoses detected     : {n_mitoses}")
    print(f"  Nottingham mitosis score: {score}/3")
    print(f"{'─'*55}")

    # Save annotated thumbnail
    save_annotated_thumbnail(
        slide, final_detections,
        str(out_dir / f"{slide_id}_annotated.png")
    )
    slide.close()

    return {
        "slide_id":          slide_id,
        "n_mitoses":         n_mitoses,
        "nottingham_score":  score,
        "detections":        final_detections,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mitosis detection — full pipeline")
    parser.add_argument("--wsi_path", required=True)
    parser.add_argument("--s1_ckpt",  default="checkpoints/stage1_best_efficientnet_b3.pth")
    parser.add_argument("--s2_ckpt",  default="checkpoints/stage2_best.pth")
    parser.add_argument("--out_dir",  default="outputs")
    args = parser.parse_args()

    result = run_pipeline(args.wsi_path, args.s1_ckpt, args.s2_ckpt, args.out_dir)
