# Crystal Highlighter — *Blumea balsamifera*
   Adaptive computer vision tool for detecting and highlighting Blumea balsamifera crystals in microscopy images using CLAHE, dual-mask (grayscale + HSV color), and watershed segmentation.


Detects and highlights *Blumea balsamifera* crystals in microscopy images using adaptive CLAHE normalization, dual grayscale/HSV masking, and watershed segmentation.

---

## Overview

Crystal Highlighter is a robust, zero-manual-tuning image processing pipeline that automatically detects and fills *Blumea balsamifera* (Sambong) crystals in microscopy images. It is designed to work across varying image types, resolutions, and lighting conditions — no per-image parameter tweaking required.

The tool processes batches of images organized in subfolders and outputs highlighted results while preserving the original folder structure.

---

## Features

- **Auto-resolution scaling** — area thresholds are fractions of total image pixels, so the same defaults work on 640×480 and 4K images alike
- **CLAHE normalization** — contrast is boosted locally before thresholding, so dark, bright, and uneven images all produce a clean binary mask
- **Dual-mask detection** — combines grayscale adaptive thresholding and HSV saturation masking to catch both colorless and yellowish-green crystals in the same image
- **Watershed splitting** — separates touching or overlapping crystals before counting and filling
- **Convex-hull fill** — replaces each contour with its convex hull before black-fill, fully covering crystal bodies including specular bright centers
- **Batch processing** — recursively processes all images and preserves the original folder structure (`A1/`, `B1/`, etc.)

---

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- tqdm

Install dependencies:

```bash
pip install opencv-python numpy tqdm
```

---

## Usage

```bash
python crystal_highlight.py --input ./data --output ./results
```

### Folder structure

```
data/
    A1/1.jpg, 2.jpg ... 12.jpg
    B1/1.jpg, 2.jpg ...
results/
    A1/1.jpg, 2.jpg ... 12.jpg   ← highlighted output
    B1/1.jpg, 2.jpg ...
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `-i` | *(required)* | Root folder containing image subfolders |
| `--output` / `-o` | `results` | Output folder |
| `--min-area-frac` | `0.000015` | Min crystal area as a fraction of image size |
| `--max-area-frac` | `0.05` | Max crystal area as a fraction of image size |
| `--min-solidity` | `0.3` | Min solidity (0.0–1.0); lower allows more irregular shapes |
| `--sat-multiplier` | `2.5` | Saturation threshold = median × this value; lower catches fainter colors |
| `--clahe-limit` | `3.0` | CLAHE clip limit |
| `--clahe-grid` | `8` | CLAHE tile grid size (N×N) |
| `--block-size` | `51` | Adaptive threshold block size (must be odd) |
| `--no-watershed` | — | Disable watershed crystal splitting |
| `--no-color` | — | Disable HSV color mask (grayscale only) |
| `--no-gray` | — | Disable grayscale mask (color mask only) |

---

## How It Works

1. **CLAHE enhancement** — each image is contrast-normalized per-tile to handle uneven lighting
2. **Grayscale mask** — adaptive Gaussian thresholding detects bright/white/colorless crystals
3. **Color mask** — two-pass HSV masking detects vivid and muted yellowish-green crystals
4. **Mask fusion** — both masks are OR-merged into a single binary mask
5. **Watershed splitting** — distance-transform watershed separates touching crystal blobs
6. **Contour filtering** — contours are filtered by area and solidity to remove noise
7. **Convex-hull fill** — each valid crystal contour is filled solid black via its convex hull

---

## Supported Image Formats

`.jpg` `.jpeg` `.png` `.tiff` `.tif` `.bmp`

---

## Example

```bash
# Run with default settings
python crystal_highlight.py --input ./data --output ./results

# Grayscale mask only, no watershed
python crystal_highlight.py --input ./data --output ./results --no-color --no-watershed

# More sensitive color detection
python crystal_highlight.py --input ./data --output ./results --sat-multiplier 1.5
```

---

## Output

After processing, the terminal prints a summary:

```
Done!
  Processed      : 48 images
  Total crystals : 312
  Output         : results/
  Folder structure preserved (A1/, A2/, B1/ ...)
```

---

## License

MIT License
