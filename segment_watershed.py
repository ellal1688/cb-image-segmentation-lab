"""
Script 2: Colony Segmentation via Watershed (scikit-image)

Segments bacterial colonies/cells in transmitted light microscopy images using:
  - Background subtraction to normalize illumination
  - Adaptive histogram equalization for contrast
  - Otsu's thresholding + morphological cleanup
  - Distance transform + peak detection for watershed markers
  - Watershed segmentation to split touching colonies
"""

import os
import csv
import glob
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes
from skimage import io, filters, morphology, segmentation, measure, exposure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "HCS Images")
TRANS_DIR = os.path.join(BASE_DIR, "trans_files")
DAPI_DIR = BASE_DIR  # DAPI images are in the main HCS Images folder
OUTPUT_DIR = os.path.join(BASE_DIR, "results_watershed")


# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_magnification(fname):
    """Extract magnification from filename (handles 10x, 10X, 20x variants)."""
    lower = fname.lower()
    if "20x" in lower:
        return "20x"
    return "10x"


def get_size_bounds(mag):
    """Return (min_area, max_area) for filtering regions by magnification."""
    if mag == "20x":
        return 500, 20000
    return 150, 8000


def get_min_distance(mag):
    """Minimum distance between watershed peaks (prevents over-segmentation)."""
    if mag == "20x":
        return 25
    return 15


# ── Segmentation pipeline ─────────────────────────────────────────────────────
def segment_image(path):
    """Run watershed segmentation on a single image.
    Returns (overlay_image, filtered_label_image, count).
    """
    fname = os.path.basename(path)
    mag = parse_magnification(fname)
    min_area, max_area = get_size_bounds(mag)
    min_dist = get_min_distance(mag)

    # 1. Load as grayscale float [0, 1]
    gray = io.imread(path, as_gray=True).astype(np.float64)

    # 2. Background subtraction
    bg = gaussian_filter(gray, sigma=50)
    subtracted = np.abs(gray - bg)

    # 3. Adaptive histogram equalization (mild to avoid boosting noise)
    enhanced = exposure.equalize_adapthist(subtracted, clip_limit=0.01)

    # 4. Smooth more aggressively to suppress background texture
    smoothed = filters.gaussian(enhanced, sigma=2.0)

    # 5. Otsu threshold — use a multiplier to be more conservative
    thresh_val = filters.threshold_otsu(smoothed)
    thresh_val = max(thresh_val * 1.2, 0.15)  # raise threshold to reject noise
    binary = smoothed > thresh_val

    # 6. Morphological cleanup
    binary = morphology.binary_opening(binary, morphology.disk(3))  # remove specks first
    binary = morphology.binary_closing(binary, morphology.disk(9))  # fill cell interiors
    binary = binary_fill_holes(binary)
    binary = morphology.remove_small_objects(binary, min_size=min_area * 2)

    # 7. Distance transform for watershed seeds
    dist = distance_transform_edt(binary)
    dist_smooth = gaussian_filter(dist, sigma=5)  # heavier smoothing to avoid over-segmentation

    # 8. Find peaks as markers
    coords = peak_local_max(dist_smooth, min_distance=min_dist, threshold_rel=0.3, labels=binary)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        markers[r, c] = i + 1
    # Dilate markers slightly so watershed has a starting region
    markers = morphology.dilation(markers, morphology.disk(2))

    # 9. Watershed
    labels = watershed(-dist_smooth, markers, mask=binary)

    # 10. Filter regions by area
    props = measure.regionprops(labels)
    valid_labels = set()
    for region in props:
        if min_area <= region.area <= max_area:
            valid_labels.add(region.label)

    # Build filtered label image
    filtered = np.where(np.isin(labels, list(valid_labels)), labels, 0)
    count = len(valid_labels)

    # 11. Build overlay
    overlay = segmentation.mark_boundaries(gray, filtered, color=(0, 1, 0), mode="thick")

    return overlay, count, gray


def count_dapi(path):
    """Count colonies in a DAPI fluorescence image (bright spots on black).
    Uses watershed to split touching nuclei. Returns count only.
    """
    fname = os.path.basename(path)
    mag = parse_magnification(fname)

    # DAPI size bounds — nuclei are smaller/rounder than full cell bodies
    if mag == "20x":
        min_area, max_area = 100, 10000
        min_dist = 20
    else:
        min_area, max_area = 30, 3000
        min_dist = 10

    # 1. Load and extract blue channel (DAPI signal) — use cv2 for robustness
    img = cv2.imread(path)
    if img is None:
        return -1
    blue = img[:, :, 0].astype(np.float64) / 255.0  # BGR format — index 0 is blue

    # 2. Gaussian blur
    smoothed = filters.gaussian(blue, sigma=1.5)

    # 3. Otsu threshold
    thresh_val = filters.threshold_otsu(smoothed)
    if thresh_val < 0.02:
        thresh_val = 0.05
    binary = smoothed > thresh_val

    # 4. Cleanup
    binary = morphology.binary_opening(binary, morphology.disk(2))
    binary = morphology.binary_closing(binary, morphology.disk(3))
    binary = binary_fill_holes(binary)
    binary = morphology.remove_small_objects(binary, min_size=min_area)

    # 5. Watershed to split touching nuclei
    dist = distance_transform_edt(binary)
    dist_smooth = gaussian_filter(dist, sigma=2)
    coords = peak_local_max(dist_smooth, min_distance=min_dist, labels=binary)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        markers[r, c] = i + 1
    markers = morphology.dilation(markers, morphology.disk(2))
    labels = watershed(-dist_smooth, markers, mask=binary)

    # 6. Filter by area
    props = measure.regionprops(labels)
    count = sum(1 for r in props if min_area <= r.area <= max_area)

    return count


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── TRANS images (full segmentation with overlays) ─────────────────────
    print("=== TRANS images (watershed) ===")
    patterns = [os.path.join(TRANS_DIR, "*.jpg"), os.path.join(TRANS_DIR, "*.JPG")]
    paths = sorted(set(p for pat in patterns for p in glob.glob(pat)))

    results = []
    for path in paths:
        fname = os.path.basename(path)
        mag = parse_magnification(fname)

        overlay, count, gray = segment_image(path)

        results.append((fname, mag, count))

        out_name = fname.rsplit(".", 1)[0] + "_overlay.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(overlay)
        ax.set_title(f"Colonies: {count}", fontsize=14)
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", dpi=100)
        plt.close(fig)

        print(f"  {fname}  mag={mag}  colonies={count}")

    csv_path = os.path.join(OUTPUT_DIR, "colony_counts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "magnification", "colony_count"])
        writer.writerows(results)

    print(f"\nTRANS done. {len(results)} images processed.")

    # ── DAPI images (counts only) ─────────────────────────────────────────
    print("\n=== DAPI images (counts only) ===")
    dapi_patterns = [
        os.path.join(DAPI_DIR, "*DAPI*.jpg"),
        os.path.join(DAPI_DIR, "*DAPI*.JPG"),
        os.path.join(DAPI_DIR, "*dapi*.jpg"),
    ]
    dapi_paths = sorted(set(p for pat in dapi_patterns for p in glob.glob(pat)))

    dapi_results = []
    for path in dapi_paths:
        fname = os.path.basename(path)
        mag = parse_magnification(fname)
        count = count_dapi(path)
        if count < 0:
            print(f"  SKIP (could not read): {fname}")
            continue
        dapi_results.append((fname, mag, count))
        print(f"  {fname}  mag={mag}  colonies={count}")

    dapi_csv = os.path.join(OUTPUT_DIR, "dapi_colony_counts.csv")
    with open(dapi_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "magnification", "colony_count"])
        writer.writerows(dapi_results)

    print(f"\nDAPI done. {len(dapi_results)} images processed.")
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
