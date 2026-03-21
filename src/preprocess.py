"""
preprocess.py
─────────────
Reads TUPAC16 / MITOS-ATYPIA-14 WSIs, applies Macenko stain normalisation,
and writes two patch datasets to disk:

  data/processed/stage1/  – 64×64 binary-class patches for the classifier
  data/processed/stage2/  – 512×512 patches + bounding-box labels for Faster R-CNN

Assumptions
-----------
- WSIs are readable by OpenSlide (.tif or .svs).
- Annotations are CSV files with (row, col) = (y, x) centroid coordinates,
  one per line. No header required.
- One CSV file per WSI, named identically (e.g. slide_001.tif → slide_001.csv).
  If a slide has no CSV, it has zero mitoses (valid, not an error).
- The train/val/test patient split is defined in a CSV with columns:
    slide_id, split          (split ∈ {train, val, test})
"""

import os
import csv
from pathlib import Path


import numpy as np
import cv2
import openslide
from sklearn.model_selection import GroupShuffleSplit

# ── Constants ──────────────────────────────────────────────────────────────────

MAGNIFICATION      = 40          # target magnification level to extract patches at
PATCH_SIZE_S1      = 64          # Stage 1 classifier patch size (pixels)
PATCH_SIZE_S2      = 512         # Stage 2 detector patch size (pixels)
STRIDE_S1          = 32          # stride for sliding window over WSI (Stage 1)
STRIDE_S2          = 256         # stride for Stage 2 patch extraction
MITOSIS_RADIUS_S1  = 32          # half-size of positive patch centred on annotation
SYNTH_BOX_HALF     = 15          # half-side of synthesised bounding box (pixels)
                                 # → box will be 30×30px inside a 512×512 patch
NEG_POS_RATIO      = 5           # how many non-mitosis patches to sample per mitosis patch

# Macenko reference target (mean optical density of a well-stained H&E slide)
# These values come from the original Macenko et al. paper and work well in practice.
MACENKO_TARGET_OD  = np.array([0.5626, 0.7201, 0.2987])


# ── Stain normalisation (Macenko method) ───────────────────────────────────────

def _rgb_to_od(patch_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 patch to optical density space."""
    patch = patch_rgb.astype(np.float32) / 255.0
    patch = np.clip(patch, 1e-6, 1.0)          # avoid log(0)
    return -np.log(patch)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Convert optical density back to RGB uint8."""
    rgb = np.exp(-od)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def macenko_normalise(patch_rgb: np.ndarray,
                      target_od: np.ndarray = MACENKO_TARGET_OD) -> np.ndarray:
    """
    Apply Macenko stain normalisation to a single RGB patch.

    This corrects for scanner-to-scanner colour variability so that a model
    trained on slides from Centre A generalises to slides from Centre B.

    Parameters
    ----------
    patch_rgb : H×W×3 uint8 ndarray
    target_od : 1-D array of length 3 — target mean optical density

    Returns
    -------
    H×W×3 uint8 ndarray — normalised patch
    """
    od = _rgb_to_od(patch_rgb)
    h, w, c = od.shape
    od_flat = od.reshape(-1, c)                 # (N, 3)

    # SVD to find the dominant stain directions
    U, S, Vt = np.linalg.svd(od_flat, full_matrices=False)
    stain_matrix = Vt[:2].T                     # (3, 2) — stain vectors (H, E)

    # Project OD values onto stain directions to get concentrations
    concentrations = od_flat @ stain_matrix     # (N, 2)

    # Normalize concentrations: scale 99th percentile to match target
    src_percentile = np.percentile(np.abs(concentrations), 99, axis=0)
    src_percentile = np.clip(src_percentile, 1e-6, None)
    tgt_percentile = np.linalg.norm(target_od[:2])
    
    scale = tgt_percentile / np.linalg.norm(src_percentile)
    
    concentrations_norm = concentrations * scale
    
    # Reconstruct OD: C * M^T where M is stain matrix
    od_norm_flat = concentrations_norm @ stain_matrix.T
    od_norm = od_norm_flat.reshape(h, w, c)

    return _od_to_rgb(od_norm)


# ── XML annotation parsing ─────────────────────────────────────────────────────

def parse_tupac16_csv(csv_path: str) -> list[tuple[int, int]]:
    """
    Parse a TUPAC16 annotation CSV and return mitosis centroids as (x, y) tuples
    in level-0 pixel coordinates.

    CSV format (no header):
        row,col     (where row=y and col=x in pixel coordinates)

    Returns a list of (x, y) tuples, or empty list if file doesn't exist.
    """
    centroids = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                y = int(float(row[0]))  # row = y
                x = int(float(row[1]))  # col = x
                centroids.append((x, y))
    except FileNotFoundError:
        pass  # No annotations for this slide (valid — zero mitoses)
    return centroids


# ── WSI patch utilities ────────────────────────────────────────────────────────

def get_level_for_magnification(slide: openslide.OpenSlide,
                                 target_mag: int = MAGNIFICATION) -> int:
    """
    Find the OpenSlide level closest to the desired magnification.
    Assumes the slide's Aperio/Leica metadata stores 'openslide.objective-power'.
    Falls back to level 0 if metadata is absent.
    """
    try:
        base_mag = int(float(slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER, target_mag)))
        downsample = base_mag / target_mag
        level = slide.get_best_level_for_downsample(downsample)
    except Exception:
        level = 0
    return level


def read_patch(slide: openslide.OpenSlide,
               x: int, y: int,
               size: int,
               level: int) -> np.ndarray:
    """
    Read a square patch from a WSI at level-0 coordinates (x, y).
    Returns an H×W×3 uint8 RGB ndarray, or None if the patch is mostly background.
    """
    region = slide.read_region((x, y), level, (size, size))
    patch  = np.array(region.convert("RGB"))

    # Reject patches that are >80% white background (tissue masking)
    grey   = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    white_ratio = np.mean(grey > 220)
    if white_ratio > 0.8:
        return None

    return patch


# ── Stage 1 patch extraction ───────────────────────────────────────────────────

def extract_stage1_patches(slide: openslide.OpenSlide,
                            centroids: list[tuple[int, int]],
                            level: int,
                            out_dir: Path,
                            slide_id: str,
                            rng: np.random.Generator) -> dict:
    """
    Extract 64×64 patches for the binary classifier.

    Positive patches  : centred on each annotated mitosis centroid.
    Negative patches  : random locations ≥ 128px from any centroid,
                        sampled at NEG_POS_RATIO × number of positives.

    Returns a dict with counts: {"pos": int, "neg": int}
    """
    pos_dir = out_dir / "mitosis"
    neg_dir = out_dir / "non_mitosis"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    half   = PATCH_SIZE_S1 // 2
    w0, h0 = slide.level_dimensions[0]     # level-0 dimensions
    counts = {"pos": 0, "neg": 0}

    # ── Positive patches ──
    for idx, (cx, cy) in enumerate(centroids):
        x0 = cx - half
        y0 = cy - half
        if x0 < 0 or y0 < 0 or x0 + PATCH_SIZE_S1 > w0 or y0 + PATCH_SIZE_S1 > h0:
            continue                        # skip patches that fall outside the slide
        patch = read_patch(slide, x0, y0, PATCH_SIZE_S1, level)
        if patch is None:
            continue
        patch = macenko_normalise(patch)
        fname = pos_dir / f"{slide_id}_pos_{idx:04d}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        counts["pos"] += 1

    # ── Negative patches (hard negatives sampled from tissue regions) ──
    n_neg_target = counts["pos"] * NEG_POS_RATIO
    centroid_arr = np.array(centroids) if centroids else np.empty((0, 2))
    attempts = 0
    neg_idx  = 0

    while neg_idx < n_neg_target and attempts < n_neg_target * 20:
        attempts += 1
        rx = int(rng.integers(half, w0 - half))
        ry = int(rng.integers(half, h0 - half))

        # Ensure this sample is far from any real mitosis
        if len(centroid_arr) > 0:
            dists = np.linalg.norm(centroid_arr - np.array([rx, ry]), axis=1)
            if dists.min() < 128:
                continue                    # too close to a real mitosis — skip

        patch = read_patch(slide, rx - half, ry - half, PATCH_SIZE_S1, level)
        if patch is None:
            continue
        patch = macenko_normalise(patch)
        fname = neg_dir / f"{slide_id}_neg_{neg_idx:04d}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        neg_idx  += 1
        counts["neg"] += 1

    return counts


# ── Stage 2 patch extraction ───────────────────────────────────────────────────

def extract_stage2_patches(slide: openslide.OpenSlide,
                            centroids: list[tuple[int, int]],
                            level: int,
                            out_dir: Path,
                            slide_id: str) -> int:
    """
    Extract 512×512 patches and write Faster R-CNN–style bounding box labels.

    A 512×512 patch is saved only if it contains at least one mitosis centroid.
    For each mitosis inside the patch a synthesised 30×30px bounding box is
    created around the centroid (since TUPAC16 provides centroids, not boxes).

    Label format (one line per box, space-separated):
        x_min  y_min  x_max  y_max  class_id
    All coordinates are in patch-local pixels. class_id = 1 for mitosis.

    Returns the number of patches written.
    """
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    half    = PATCH_SIZE_S2 // 2
    w0, h0  = slide.level_dimensions[0]
    written = 0

    # Build a set of 512×512 patch origins that contain at least one centroid
    # Strategy: for each centroid, emit a patch centred on it (clamped to slide bounds)
    seen_origins = set()

    for cx, cy in centroids:
        # Clamp patch origin so it stays inside the slide
        ox = int(np.clip(cx - half, 0, w0 - PATCH_SIZE_S2))
        oy = int(np.clip(cy - half, 0, h0 - PATCH_SIZE_S2))
        seen_origins.add((ox, oy))

    for patch_idx, (ox, oy) in enumerate(seen_origins):
        patch = read_patch(slide, ox, oy, PATCH_SIZE_S2, level)
        if patch is None:
            continue
        patch = macenko_normalise(patch)

        # Find all centroids that fall inside this patch
        boxes = []
        for cx, cy in centroids:
            lx = cx - ox           # local x coordinate inside patch
            ly = cy - oy           # local y coordinate inside patch
            if 0 <= lx < PATCH_SIZE_S2 and 0 <= ly < PATCH_SIZE_S2:
                x_min = max(0,             lx - SYNTH_BOX_HALF)
                y_min = max(0,             ly - SYNTH_BOX_HALF)
                x_max = min(PATCH_SIZE_S2, lx + SYNTH_BOX_HALF)
                y_max = min(PATCH_SIZE_S2, ly + SYNTH_BOX_HALF)
                boxes.append((x_min, y_min, x_max, y_max, 1))  # class_id = 1

        if not boxes:
            continue                # no centroids landed inside — skip

        stem  = f"{slide_id}_s2_{patch_idx:04d}"
        cv2.imwrite(str(img_dir / f"{stem}.png"),
                    cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

        with open(lbl_dir / f"{stem}.txt", "w") as f:
            for x_min, y_min, x_max, y_max, cls in boxes:
                f.write(f"{x_min} {y_min} {x_max} {y_max} {cls}\n")

        written += 1

    return written


# ── Patient-level split ────────────────────────────────────────────────────────

def build_patient_split(slide_ids: list[str],
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         random_state: int = 42) -> dict[str, str]:
    """
    Assign each slide_id to train / val / test so that no patient's slides
    appear in more than one split.

    TUPAC16 uses one slide per patient so slide_id == patient_id here.
    If your dataset has multiple slides per patient, pass patient IDs as
    `groups` and adjust accordingly.

    Returns a dict  {slide_id: "train" | "val" | "test"}
    """
    ids    = np.array(slide_ids)
    groups = np.arange(len(ids))            # one patient per slide for TUPAC16

    splitter = GroupShuffleSplit(n_splits=1,
                                  test_size=test_ratio,
                                  random_state=random_state)
    trainval_idx, test_idx = next(splitter.split(ids, groups=groups))

    splitter2 = GroupShuffleSplit(n_splits=1,
                                   test_size=val_ratio / (1 - test_ratio),
                                   random_state=random_state)
    train_idx, val_idx = next(splitter2.split(
        ids[trainval_idx], groups=groups[trainval_idx]))

    split_map = {}
    for i in ids[trainval_idx][train_idx]: split_map[i] = "train"
    for i in ids[trainval_idx][val_idx]:   split_map[i] = "val"
    for i in ids[test_idx]:                split_map[i] = "test"
    return split_map


# ── Main driver ────────────────────────────────────────────────────────────────

def run_preprocessing(raw_wsi_dir:  str,
                       raw_ann_dir:  str,
                       out_base_dir: str,
                       random_state: int = 42) -> None:
    """
    Full preprocessing run over all WSIs found in raw_wsi_dir.

    Parameters
    ----------
    raw_wsi_dir  : directory containing .tif / .svs files
    raw_ann_dir  : directory containing matching .csv annotation files
    out_base_dir : root output directory (data/processed/)
    """
    rng      = np.random.default_rng(random_state)
    wsi_dir  = Path(raw_wsi_dir)
    ann_dir  = Path(raw_ann_dir)
    out_base = Path(out_base_dir)

    # Handle both flat and nested directory structures (e.g., 01/01.tif)
    wsi_paths  = sorted(list(wsi_dir.glob("**/*.tif")) + list(wsi_dir.glob("**/*.svs")))
    slide_ids  = [p.stem for p in wsi_paths]

    if not slide_ids:
        raise FileNotFoundError(f"No .tif or .svs files found in {raw_wsi_dir} (recursive)")

    # Build patient-level split before extracting any patches
    split_map = build_patient_split(slide_ids)

    # Write the split map to CSV so training scripts can use it
    split_csv = out_base / "split.csv"
    out_base.mkdir(parents=True, exist_ok=True)
    with open(split_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_id", "split"])
        for sid, sp in split_map.items():
            writer.writerow([sid, sp])
    print(f"Split map written → {split_csv}")

    total_s1_pos = total_s1_neg = total_s2 = 0

    for wsi_path in wsi_paths:
        slide_id = wsi_path.stem
        # Look for CSV in the same nested directory as the TIFF
        # E.g., if wsi_path is wsis/mitoses_image_data_part_1/01/01.tif,
        # look for ann_dir/mitoses_ground_truth/01/01.csv
        parent_dir = wsi_path.parent.name  # e.g., "01"
        csv_path = ann_dir / "mitoses_ground_truth" / parent_dir / f"{slide_id}.csv"
        if not csv_path.exists():
            # Fallback: try direct lookup
            csv_path = ann_dir / f"{slide_id}.csv"
        
        split    = split_map.get(slide_id, "train")

        print(f"\n[{split.upper()}] Processing {slide_id} …")

        # Parse centroids from CSV
        centroids = parse_tupac16_csv(str(csv_path))
        if not centroids:
            print(f"  No annotation file found for {slide_id} (zero mitoses). "
                  "Extracting background patches only.")

        try:
            slide = openslide.open_slide(str(wsi_path))
        except Exception as e:
            print(f"  ERROR opening slide: {e}. Skipping.")
            continue

        level = get_level_for_magnification(slide)
        print(f"  Magnification level: {level}  "
              f"  Dimensions: {slide.level_dimensions[level]}")

        # Stage 1 patches
        s1_out  = out_base / "stage1" / split
        counts  = extract_stage1_patches(slide, centroids, level, s1_out,
                                          slide_id, rng)
        print(f"  Stage 1 → {counts['pos']} positive, {counts['neg']} negative patches")
        total_s1_pos += counts["pos"]
        total_s1_neg += counts["neg"]

        # Stage 2 patches (only from train — val/test use slide-level inference)
        if split == "train":
            s2_out = out_base / "stage2" / split
            n_s2   = extract_stage2_patches(slide, centroids, level, s2_out, slide_id)
            print(f"  Stage 2 → {n_s2} detection patches")
            total_s2 += n_s2

        slide.close()

    print(f"\n{'─'*55}")
    print(f"Preprocessing complete.")
    print(f"  Stage 1 total: {total_s1_pos} positive, {total_s1_neg} negative")
    print(f"  Stage 2 total: {total_s2} detection patches (train only)")
    print(f"  Split CSV    : {split_csv}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mitosis detection — preprocessing")
    parser.add_argument("--wsi_dir", required=True, help="Raw WSI directory")
    parser.add_argument("--ann_dir", required=True, help="Raw annotation XML directory")
    parser.add_argument("--out_dir", default="data/processed", help="Output base directory")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    run_preprocessing(args.wsi_dir, args.ann_dir, args.out_dir, args.seed)
