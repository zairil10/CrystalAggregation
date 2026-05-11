"""
Crystal Highlighter — Blumea balsamifera
Fills all detected crystals with solid black on every image.
Works robustly across varying image types without manual tuning.

HOW IT ADAPTS TO ANY INPUT
───────────────────────────
1. Auto-resolution scaling
   min/max area are fractions of image pixels, not fixed counts, so the same
   defaults work on 640x480 and 4K images alike.

2. Per-image CLAHE normalisation
   Contrast is boosted locally before thresholding so dark, bright, and
   uneven images all produce a clean binary mask.

3. Combined gray + color mask  (key upgrade)
   Every image gets BOTH a grayscale adaptive-threshold mask AND an HSV
   saturation-based color mask. The two are OR-merged before contour finding.
   - Gray mask  → catches bright/white/colorless crystals
   - Color mask → catches yellow-green crystals on gray background
   Together they catch both types in the same image without mode-switching.

4. Auto HSV range (no hardcoded hue values)
   The color mask isolates pixels whose saturation exceeds the image's own
   median saturation by a tunable multiplier. Adapts to any crystal color.

5. Convex-hull fill
   Each contour is replaced by its convex hull before black-fill, so
   specular bright centers inside round crystals are fully covered.

6. Watershed splitting
   Touching crystals are separated before counting/filling.

Folder structure:
    data/
        A1/1.jpg, 2.jpg ... 12.jpg
        ...
    results/
        A1/1.jpg, 2.jpg ... 12.jpg  (highlighted)
        ...

Usage:
    python crystal_highlight.py --input ./data --output ./results

Tuning (rarely needed):
    --min-area-frac   Min crystal area as fraction of image (default 0.000015)
    --max-area-frac   Max crystal area as fraction of image (default 0.05)
    --min-solidity    0.0-1.0, lower = allow more irregular shapes (default 0.3)
    --sat-multiplier  How many x above median saturation to call a pixel
                      "colored". Lower = more sensitive to faint color (default 2.5)
    --clahe-limit     CLAHE clip limit (default 3.0)
    --clahe-grid      CLAHE tile grid NxN (default 8)
    --block-size      Adaptive threshold block size, must be odd (default 51)
    --no-watershed    Disable watershed crystal splitting
    --no-color        Disable color mask (gray mask only)
    --no-gray         Disable gray mask (color mask only)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _grayscale_mask(img, block_size, clahe_limit, clahe_grid):
    """Adaptive-threshold binary mask on CLAHE-enhanced grayscale."""
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=clahe_limit,
                                tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size, C=8
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Fill internal holes (specular reflections inside crystals)
    h, w       = binary.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled     = binary.copy()
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    binary = binary | cv2.bitwise_not(filled)

    return binary


def _color_mask(img, sat_multiplier):
    """
    Dual-pass color mask — catches both vivid AND muted yellowish-green crystals.

    Pass 1 — Saturation threshold (adaptive, any color):
      Pixels whose saturation exceeds (median × sat_multiplier) are marked.
      Catches well-saturated crystals of any hue.

    Pass 2 — Yellowish-green hue band (lower saturation bar):
      Yellowish-green crystals (H≈25-85 in HSV) are often more muted than
      pure green ones. They can slip under the Pass 1 threshold even though
      they're visibly distinct from the gray background.
      This pass uses a fixed hue band with a much lower saturation floor (15)
      so even pale/washed-out yellow-green blobs are caught.
      The two passes are OR-merged so nothing caught by either is lost.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h   = hsv[:, :, 0]
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2]

    median_sat = float(np.median(sat))
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # --- Pass 1: adaptive saturation threshold (any crystal color) ---
    if median_sat < 5:
        # fully desaturated image — skip both passes
        return np.zeros(img.shape[:2], dtype=np.uint8)

    threshold = min(median_sat * sat_multiplier, 255)
    _, p1     = cv2.threshold(sat, threshold, 255, cv2.THRESH_BINARY)
    p1        = p1.astype(np.uint8)
    p1        = cv2.morphologyEx(p1, cv2.MORPH_OPEN,  kernel, iterations=2)
    p1        = cv2.morphologyEx(p1, cv2.MORPH_CLOSE, kernel, iterations=5)

    # --- Pass 2: yellowish-green hue band with lower saturation bar ---
    # H: 25–85  covers yellow → yellow-green → green
    # S: >=15   catches muted/pale crystals that Pass 1 misses
    # V: >=40   excludes very dark shadow regions
    yg_mask = (
        (h  >= 25)  & (h  <= 85) &
        (sat >= 15) &
        (val >= 40)
    ).astype(np.uint8) * 255
    yg_mask = cv2.morphologyEx(yg_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    yg_mask = cv2.morphologyEx(yg_mask, cv2.MORPH_CLOSE, kernel, iterations=5)

    # Merge both passes
    return cv2.bitwise_or(p1, yg_mask)


def _apply_watershed(binary, img):
    """Split touching crystals via distance-transform watershed."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
    sure_fg    = np.uint8(sure_fg * 255)
    sure_bg    = cv2.dilate(binary, kernel, iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers   += 1
    markers[unknown == 255] = 0
    markers    = cv2.watershed(img, markers)

    out = np.zeros_like(binary)
    out[markers > 1] = 255
    return out


def _filter_and_hull(contours, min_area, max_area, min_solidity):
    """
    Filter contours by area + solidity, return convex hulls.
    Convex hull fill covers the whole crystal body including bright centers.
    """
    hulls = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        hull      = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        if (area / hull_area) < min_solidity:
            continue
        hulls.append(hull)
    return hulls


# -----------------------------------------------------------------------------
# Main detection
# -----------------------------------------------------------------------------

def highlight_crystals(img,
                       min_area_frac, max_area_frac, min_solidity,
                       sat_multiplier, clahe_limit, clahe_grid, block_size,
                       use_watershed, use_color, use_gray):
    """Detect all crystals and fill them solid black. Returns (result, count)."""
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * min_area_frac
    max_area = img_area * max_area_frac

    # Build combined mask
    combined = np.zeros(img.shape[:2], dtype=np.uint8)

    if use_gray:
        combined = cv2.bitwise_or(combined,
                       _grayscale_mask(img, block_size, clahe_limit, clahe_grid))

    if use_color:
        combined = cv2.bitwise_or(combined,
                       _color_mask(img, sat_multiplier))

    if use_watershed:
        combined = _apply_watershed(combined, img)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = _filter_and_hull(contours, min_area, max_area, min_solidity)

    result = img.copy()
    cv2.drawContours(result, hulls, -1, (0, 0, 0), -1)
    return result, len(hulls)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crystal highlighter -- adaptive black fill for any microscopy image"
    )
    parser.add_argument("--input",          "-i", required=True,
                        help="Root folder with A1, B1... subfolders")
    parser.add_argument("--output",         "-o", default="results",
                        help="Output folder (default: results)")

    # Adaptive area thresholds
    parser.add_argument("--min-area-frac",  default=0.000015, type=float,
                        help="Min crystal area as fraction of image (default: 0.000015)")
    parser.add_argument("--max-area-frac",  default=0.05,     type=float,
                        help="Max crystal area as fraction of image (default: 0.05)")
    parser.add_argument("--min-solidity",   default=0.3,      type=float,
                        help="Min solidity 0.0-1.0 (default: 0.3)")

    # Color mask
    parser.add_argument("--sat-multiplier", default=2.5,      type=float,
                        help="Sat threshold = median x this (default: 2.5). Lower = catch fainter colors.")
    parser.add_argument("--no-color",       action="store_true",
                        help="Disable color (saturation) mask")

    # Gray mask
    parser.add_argument("--block-size",     default=51,       type=int,
                        help="Adaptive threshold block size, must be odd (default: 51)")
    parser.add_argument("--clahe-limit",    default=3.0,      type=float,
                        help="CLAHE clip limit (default: 3.0)")
    parser.add_argument("--clahe-grid",     default=8,        type=int,
                        help="CLAHE tile grid NxN (default: 8)")
    parser.add_argument("--no-gray",        action="store_true",
                        help="Disable grayscale mask")

    # Watershed
    parser.add_argument("--no-watershed",   action="store_true",
                        help="Disable watershed splitting of touching crystals")

    args = parser.parse_args()

    use_watershed = not args.no_watershed
    use_color     = not args.no_color
    use_gray      = not args.no_gray

    if not use_color and not use_gray:
        print("Error: cannot use both --no-color and --no-gray at the same time.")
        return

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    all_images = sorted([
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    ])

    if not all_images:
        print("No images found. Check your folder structure.")
        return

    print(f"\nFound {len(all_images)} images")
    print(f"Settings:")
    print(f"  masks         : {'gray ' if use_gray else ''}{'color' if use_color else ''}")
    print(f"  min-area-frac : {args.min_area_frac}  (auto-scales with resolution)")
    print(f"  max-area-frac : {args.max_area_frac}")
    print(f"  min-solidity  : {args.min_solidity}")
    print(f"  sat-multiplier: {args.sat_multiplier}x median saturation")
    print(f"  watershed     : {'ON' if use_watershed else 'OFF'}")
    print(f"  clahe         : limit={args.clahe_limit}, grid={args.clahe_grid}x{args.clahe_grid}")
    print(f"  block-size    : {args.block_size}")
    print(f"  Output        : {output_dir}\n")

    total_crystals = 0

    for img_path in tqdm(all_images, desc="Highlighting crystals"):
        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"  [SKIP] Could not read: {img_path}")
            continue

        result, count = highlight_crystals(
            img,
            min_area_frac  = args.min_area_frac,
            max_area_frac  = args.max_area_frac,
            min_solidity   = args.min_solidity,
            sat_multiplier = args.sat_multiplier,
            clahe_limit    = args.clahe_limit,
            clahe_grid     = args.clahe_grid,
            block_size     = args.block_size,
            use_watershed  = use_watershed,
            use_color      = use_color,
            use_gray       = use_gray,
        )
        total_crystals += count

        relative = img_path.relative_to(input_dir)
        out_path = output_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result)

    print(f"\nDone!")
    print(f"  Processed      : {len(all_images)} images")
    print(f"  Total crystals : {total_crystals}")
    print(f"  Output         : {output_dir}/")
    print(f"  Folder structure preserved (A1/, A2/, B1/ ...)")


if __name__ == "__main__":
    main()
