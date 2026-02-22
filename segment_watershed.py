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
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes
from skimage import io, filters, morphology, segmentation, measure, exposure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_DIR = os.path.join(os.path.dirname(__file__), "HCS Images", "trans_files")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "HCS Images", "results_watershed")


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


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Gather all TRANS jpg files (case-insensitive)
    patterns = [os.path.join(INPUT_DIR, "*.jpg"), os.path.join(INPUT_DIR, "*.JPG")]
    paths = sorted(set(p for pat in patterns for p in glob.glob(pat)))

    results = []
    for path in paths:
        fname = os.path.basename(path)
        mag = parse_magnification(fname)

        overlay, count, gray = segment_image(path)

        results.append((fname, mag, count))

        # Save overlay using matplotlib (preserves float image correctly)
        out_name = fname.rsplit(".", 1)[0] + "_overlay.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(overlay)
        ax.set_title(f"Colonies: {count}", fontsize=14)
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", dpi=100)
        plt.close(fig)

        print(f"  {fname}  mag={mag}  colonies={count}")

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "colony_counts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "magnification", "colony_count"])
        writer.writerows(results)

    print(f"\nDone. {len(results)} images processed.")
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
