# Colony Segmentation Lab

Two Python scripts for segmenting bacterial colonies/cells in transmitted light (TRANS) microscopy images using different approaches.

## Setup

### 1. Install dependencies

```bash
pip3 install opencv-python scikit-image matplotlib Pillow numpy scipy
```

### 2. Add your images

Place your TRANS `.jpg` images in:

```
HCS Images/trans_files/
```

Expected filename format: `G7_20x_C2_F1_T0_TRANS.jpg` (group, magnification, well, field, timepoint).

## Running the scripts

### Script 1: Thresholding + Contour Detection (OpenCV)

```bash
python3 segment_threshold.py
```

**How it works:**
1. Background subtraction (Gaussian blur) to normalize illumination
2. CLAHE contrast enhancement
3. Otsu's automatic thresholding
4. Morphological close/open to fill cell interiors and remove noise
5. Contour detection with area-based filtering

**Output:** `HCS Images/results_threshold/`

### Script 2: Watershed Segmentation (scikit-image)

```bash
python3 segment_watershed.py
```

**How it works:**
1. Background subtraction + adaptive histogram equalization
2. Otsu's thresholding with morphological cleanup
3. Distance transform to find cell centers
4. Watershed algorithm to split touching cells along boundaries

**Output:** `HCS Images/results_watershed/`

## Output

Each script produces:
- **Overlay images** — original image with green colony boundaries drawn on top
- **`colony_counts.csv`** — table with `filename, magnification, colony_count` per image

## Notes

- Both scripts auto-detect magnification (10x vs 20x) from filenames and adjust size thresholds accordingly
- The threshold approach is more sensitive (detects more cells, but also more noise)
- The watershed approach produces cleaner borders and better separates touching cells
