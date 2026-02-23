"""
Script 1: Colony Segmentation via Thresholding + Contour Detection (OpenCV)

Segments bacterial colonies/cells in transmitted light microscopy images using:
  - Background subtraction to normalize illumination
  - CLAHE for contrast enhancement
  - Otsu's thresholding
  - Morphological operations to clean up and fill ring/donut shapes
  - Contour detection with area filtering
"""

import os
import csv
import glob
import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "HCS Images")
TRANS_DIR = os.path.join(BASE_DIR, "trans_files")
DAPI_DIR = BASE_DIR  # DAPI images are in the main HCS Images folder
OUTPUT_DIR = os.path.join(BASE_DIR, "results_threshold")


# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_magnification(fname):
    """Extract magnification from filename (handles 10x, 10X, 20x variants)."""
    lower = fname.lower()
    if "20x" in lower:
        return "20x"
    return "10x"


def get_size_bounds(mag):
    """Return (min_area, max_area) for filtering contours by magnification."""
    if mag == "20x":
        return 500, 20000
    return 150, 8000


# ── Segmentation pipeline ─────────────────────────────────────────────────────
def segment_image(path):
    """Run thresholding + contour segmentation on a single image.
    Returns (overlay_image, list_of_contours).
    """
    fname = os.path.basename(path)
    mag = parse_magnification(fname)
    min_area, max_area = get_size_bounds(mag)

    # 1. Load as grayscale
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, []

    # 2. Background subtraction – remove uneven illumination
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=50)
    # Capture both bright-on-dark and dark-on-bright features
    bright = cv2.subtract(gray, bg)
    dark = cv2.subtract(bg, gray)
    combined = cv2.add(bright, dark)

    # 3. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)

    # 4. Light blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 5. Otsu's threshold
    thresh_val, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Fallback if Otsu picks a very low threshold (sparse images)
    if thresh_val < 20:
        _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # 6. Morphological close (fill donut interiors) then open (remove specks)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # 7. Fill holes by drawing filled external contours
    ext_contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(opened)
    cv2.drawContours(filled, ext_contours, -1, 255, thickness=-1)

    # 8. Final contour detection + size filtering
    contours, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    # 9. Build overlay
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, valid, -1, (0, 255, 0), 2)
    label = f"Count: {len(valid)}"
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return overlay, valid


def count_dapi(path):
    """Count colonies in a DAPI fluorescence image (bright spots on black).
    Returns colony count only (no overlay).
    """
    fname = os.path.basename(path)
    mag = parse_magnification(fname)

    # DAPI size bounds — nuclei are smaller/rounder than full cell bodies
    if mag == "20x":
        min_area, max_area = 100, 10000
    else:
        min_area, max_area = 30, 3000

    # 1. Load color and extract blue channel (DAPI signal)
    img = cv2.imread(path)
    if img is None:
        return -1
    blue = img[:, :, 0]  # BGR format — index 0 is blue

    # 2. Gaussian blur to smooth noise
    blurred = cv2.GaussianBlur(blue, (5, 5), 0)

    # 3. Otsu's threshold — bright nuclei on dark background
    thresh_val, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if thresh_val < 10:
        _, binary = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    # 4. Morphological open to remove noise, close to fill gaps
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # 5. Find contours and filter by area
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    return len(valid)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── TRANS images (full segmentation with overlays) ─────────────────────
    print("=== TRANS images (thresholding + contours) ===")
    patterns = [os.path.join(TRANS_DIR, "*.jpg"), os.path.join(TRANS_DIR, "*.JPG")]
    paths = sorted(set(p for pat in patterns for p in glob.glob(pat)))

    results = []
    for path in paths:
        fname = os.path.basename(path)
        mag = parse_magnification(fname)
        overlay, contours = segment_image(path)
        if overlay is None:
            print(f"  SKIP (could not read): {fname}")
            continue

        count = len(contours)
        results.append((fname, mag, count))

        out_name = fname.rsplit(".", 1)[0] + "_overlay.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), overlay)
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
