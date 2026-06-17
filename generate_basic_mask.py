#!/usr/bin/env python3
"""Generate a first-pass detector bad-pixel mask from one TIFF image.

The workflow is intentionally dependency-light:
  1. read a TIFF image with Pillow
  2. make a percentile-clipped log-normalized intensity image
  3. compute local median, local residual, and robust local z-score
  4. threshold the local z-score to create a basic bad-pixel mask

The output mask uses 1 for bad/invalid pixels and 0 for good pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}


def find_first_tiff(folder: Path) -> Path:
    candidates = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix in TIFF_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No TIFF files found in {folder}")
    return candidates[0]


def read_tiff(path: Path, frame: int = 0) -> np.ndarray:
    with Image.open(path) as image:
        if frame:
            image.seek(frame)
        array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D detector image, got shape {array.shape}")
    return array


def finite_percentile(array: np.ndarray, q: float) -> float:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels")
    return float(np.percentile(finite, q))


def normalize_log_intensity(
    image: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Shift, log-scale, and normalize an image to approximately [0, 1]."""

    finite = np.isfinite(image)
    low = finite_percentile(image, low_percentile)

    shifted = np.zeros_like(image, dtype=np.float32)
    shifted[finite] = np.maximum(image[finite] - low, 0.0)
    log_image = np.log1p(shifted, dtype=np.float32)

    high = finite_percentile(log_image, high_percentile)
    if high <= 0:
        high = float(np.max(log_image[finite]))
    if high <= 0:
        raise ValueError("Could not find a positive normalization scale")

    normalized = np.clip(log_image / high, 0.0, 1.0).astype(np.float32)
    normalized[~finite] = np.nan

    return normalized, {
        "raw_low_percentile_value": low,
        "log_high_percentile_value": high,
    }


def median_filter_tiled(array: np.ndarray, window: int, tile_rows: int) -> np.ndarray:
    """Median filter using reflected edges and row tiles to limit memory use."""

    if window % 2 != 1 or window < 3:
        raise ValueError("--window must be an odd integer >= 3")

    pad = window // 2
    height, width = array.shape
    out = np.empty_like(array, dtype=np.float32)

    finite = np.isfinite(array)
    fill_value = float(np.median(array[finite])) if finite.any() else 0.0
    clean = np.where(finite, array, fill_value).astype(np.float32, copy=False)
    padded = np.pad(clean, pad_width=pad, mode="reflect")

    for row0 in range(0, height, tile_rows):
        row1 = min(row0 + tile_rows, height)
        block = padded[row0 : row1 + 2 * pad, :]
        windows = sliding_window_view(block, (window, window))
        out[row0:row1, :] = np.median(windows, axis=(-2, -1)).astype(np.float32)

    return out


def robust_local_zscore(
    analysis_image: np.ndarray,
    window: int,
    tile_rows: int,
    sigma_floor_percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    local_median = median_filter_tiled(analysis_image, window, tile_rows)
    residual = analysis_image - local_median

    abs_residual = np.abs(residual)
    local_mad = median_filter_tiled(abs_residual, window, tile_rows)
    robust_sigma = (1.4826 * local_mad).astype(np.float32)

    positive_sigma = robust_sigma[np.isfinite(robust_sigma) & (robust_sigma > 0)]
    if positive_sigma.size:
        sigma_floor = float(np.percentile(positive_sigma, sigma_floor_percentile))
    else:
        sigma_floor = 1.0e-6
    sigma_floor = max(sigma_floor, 1.0e-6)

    zscore = residual / np.maximum(robust_sigma, sigma_floor)
    zscore = zscore.astype(np.float32)
    return local_median, residual.astype(np.float32), zscore, sigma_floor


def binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = expanded
    return result


def save_mask_tiff(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8)).save(path)


def as_uint8_grayscale(array: np.ndarray, low_q: float = 0.5, high_q: float = 99.5) -> np.ndarray:
    finite = np.isfinite(array)
    low = float(np.percentile(array[finite], low_q))
    high = float(np.percentile(array[finite], high_q))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return (255.0 * scaled).astype(np.uint8)


def save_grayscale_preview(array: np.ndarray, path: Path) -> None:
    Image.fromarray(as_uint8_grayscale(array)).save(path)


def save_diverging_preview(array: np.ndarray, path: Path, percentile: float = 99.0) -> None:
    finite = np.isfinite(array)
    scale = float(np.percentile(np.abs(array[finite]), percentile))
    if scale <= 0:
        scale = 1.0

    normalized = np.clip(array / scale, -1.0, 1.0)
    normalized[~finite] = 0.0
    magnitude = np.abs(normalized)
    base = 255.0 * (1.0 - magnitude)

    red = base + 255.0 * np.clip(normalized, 0.0, 1.0)
    green = base
    blue = base + 255.0 * np.clip(-normalized, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def save_mask_overlay(base_image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    gray = as_uint8_grayscale(base_image)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    red = np.array([255, 0, 0], dtype=np.uint8)
    rgb[mask] = (0.35 * rgb[mask] + 0.65 * red).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a statistical bad-pixel mask from a detector TIFF image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Input TIFF file. If omitted, the first TIFF in the current folder is used.",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index for multi-page TIFFs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mask_baseline_output"),
        help="Directory for masks, arrays, and previews.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Odd local neighborhood size for median/MAD calculations.",
    )
    parser.add_argument(
        "--tile-rows",
        type=int,
        default=256,
        help="Rows processed at once during tiled median filtering.",
    )
    parser.add_argument(
        "--log-low-percentile",
        type=float,
        default=0.5,
        help="Raw intensity percentile used as the zero point before log scaling.",
    )
    parser.add_argument(
        "--log-high-percentile",
        type=float,
        default=99.5,
        help="Log intensity percentile mapped to 1.0 after log scaling.",
    )
    parser.add_argument(
        "--dead-z",
        type=float,
        default=-8.0,
        help="Pixels with local z-score <= this value are marked bad.",
    )
    parser.add_argument(
        "--hot-z",
        type=float,
        default=12.0,
        help="Pixels with local z-score >= this value are marked bad.",
    )
    parser.add_argument(
        "--sigma-floor-percentile",
        type=float,
        default=5.0,
        help="Percentile floor for local robust sigma to avoid division by tiny noise.",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=0,
        help="Optional dilation iterations applied to the final mask.",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=0,
        help="Optional border width, in pixels, always marked bad.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    image_path = args.image if args.image is not None else find_first_tiff(Path.cwd())
    image_path = image_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_tiff(image_path, frame=args.frame)
    finite = np.isfinite(raw)

    log_normalized, log_info = normalize_log_intensity(
        raw,
        low_percentile=args.log_low_percentile,
        high_percentile=args.log_high_percentile,
    )

    local_median, local_residual, local_zscore, sigma_floor = robust_local_zscore(
        log_normalized,
        window=args.window,
        tile_rows=args.tile_rows,
        sigma_floor_percentile=args.sigma_floor_percentile,
    )

    invalid_mask = ~finite
    dead_mask = finite & (local_zscore <= args.dead_z)
    hot_mask = finite & (local_zscore >= args.hot_z)
    mask = invalid_mask | dead_mask | hot_mask

    if args.border > 0:
        b = args.border
        mask[:b, :] = True
        mask[-b:, :] = True
        mask[:, :b] = True
        mask[:, -b:] = True

    if args.dilate > 0:
        mask = binary_dilate(mask, args.dilate)

    stem = image_path.stem
    mask_path = output_dir / f"{stem}_basic_bad_pixel_mask.tiff"
    dead_mask_path = output_dir / f"{stem}_dead_candidate_mask.tiff"
    hot_mask_path = output_dir / f"{stem}_hot_candidate_mask.tiff"
    log_path = output_dir / f"{stem}_log_normalized.npy"
    median_path = output_dir / f"{stem}_local_median.npy"
    residual_path = output_dir / f"{stem}_local_residual.npy"
    zscore_path = output_dir / f"{stem}_local_zscore.npy"
    summary_path = output_dir / f"{stem}_summary.json"

    save_mask_tiff(mask, mask_path)
    save_mask_tiff(dead_mask, dead_mask_path)
    save_mask_tiff(hot_mask, hot_mask_path)
    np.save(log_path, log_normalized)
    np.save(median_path, local_median)
    np.save(residual_path, local_residual)
    np.save(zscore_path, local_zscore)

    save_grayscale_preview(log_normalized, output_dir / f"{stem}_preview_log_normalized.png")
    save_grayscale_preview(local_median, output_dir / f"{stem}_preview_local_median.png")
    save_diverging_preview(local_residual, output_dir / f"{stem}_preview_local_residual.png")
    save_diverging_preview(local_zscore, output_dir / f"{stem}_preview_local_zscore.png")
    save_mask_overlay(log_normalized, mask, output_dir / f"{stem}_preview_mask_overlay.png")

    finite_raw = raw[finite]
    bad_count = int(mask.sum())
    total_count = int(mask.size)
    summary = {
        "input_image": str(image_path),
        "shape": list(raw.shape),
        "dtype_after_read": str(raw.dtype),
        "raw_min": float(np.min(finite_raw)),
        "raw_max": float(np.max(finite_raw)),
        "raw_mean": float(np.mean(finite_raw)),
        "raw_median": float(np.median(finite_raw)),
        "log_low_percentile": args.log_low_percentile,
        "log_high_percentile": args.log_high_percentile,
        **log_info,
        "local_window": args.window,
        "dead_z_threshold": args.dead_z,
        "hot_z_threshold": args.hot_z,
        "sigma_floor": sigma_floor,
        "dilate_iterations": args.dilate,
        "border_pixels_marked_bad": args.border,
        "invalid_pixel_count": int(invalid_mask.sum()),
        "dead_candidate_count_before_dilation_or_border": int(dead_mask.sum()),
        "hot_candidate_count_before_dilation_or_border": int(hot_mask.sum()),
        "bad_pixel_count": bad_count,
        "total_pixel_count": total_count,
        "bad_pixel_fraction": bad_count / total_count,
        "outputs": {
            "mask_tiff": str(mask_path),
            "dead_candidate_mask_tiff": str(dead_mask_path),
            "hot_candidate_mask_tiff": str(hot_mask_path),
            "log_normalized_npy": str(log_path),
            "local_median_npy": str(median_path),
            "local_residual_npy": str(residual_path),
            "local_zscore_npy": str(zscore_path),
        },
    }
    write_summary(summary_path, summary)

    print(f"Input: {image_path}")
    print(f"Mask: {mask_path}")
    print(f"Bad pixels: {bad_count} / {total_count} ({100.0 * bad_count / total_count:.4f}%)")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
