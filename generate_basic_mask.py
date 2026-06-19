"""Class-based detector bad-pixel mask generation.

The generated mask uses 1 for bad/invalid pixels and 0 for good pixels.
The class can start from a user-supplied baseline .npy mask, then add:
  1. invalid/non-finite pixels
  2. local low/high outliers from a robust local z-score
  3. an optional detector border mask
  4. the existing optional beamstop detector

No files are written automatically. Import the class, tune parameters when calling
``generate_mask()``, and call ``save_mask_npy()`` or ``save_summary_json()`` only
when you want output files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}


class DetectorMaskGenerator:
    """Generate detector masks from a TIFF image and an optional baseline .npy mask.

    Parameters can be set at initialization and overridden per call to
    ``generate_mask``. The final returned mask is a uint8 array with values
    0 for good pixels and 1 for bad pixels.
    """

    LEGACY_PARAMETER_ALIASES = {
        "window": "zscore_window",
        "tile_rows": "zscore_tile_rows",
    }

    DEFAULTS: dict[str, Any] = {
        # TIFF frame to read for multi-page TIFFs; ordinary single-frame TIFFs use 0.
        "frame": 0,
        # If True, robust_local_zscore subtracts a tiled local median and uses a
        # tiled local MAD noise estimate. If False, it uses a global robust
        # median/MAD z-score, which can preserve sharp hot-pixel spikes better
        # but is more sensitive to broad intensity gradients.
        "use_median_filter": True,
        # Odd local neighborhood width for robust_local_zscore's local
        # median/MAD calculation. Larger windows are smoother but less sensitive
        # to narrow detector defects.
        "zscore_window": 7,
        # Number of detector rows processed per tile inside robust_local_zscore.
        # Lower this if memory becomes tight for very large detectors.
        "zscore_tile_rows": 256,
        # Independent defaults for calling median_filter_tiled directly. These
        # do not affect robust_local_zscore unless you pass them there yourself.
        "median_window": 7,
        "median_tile_rows": 256,
        # Raw intensity percentile treated as the zero point before log scaling.
        # Raising this suppresses more low-end background before analysis.
        "log_low_percentile": 0.5,
        # Log intensity percentile mapped to 1.0. Lower values increase contrast
        # in bright regions but can compress very intense scattering.
        "log_high_percentile": 99.5,
        # Local z-score thresholds for low-response/dead and high-response/hot
        # candidates. More extreme values make the mask more conservative.
        "dead_z": -8.0,
        "hot_z": 12.0,
        # Percentile floor for local robust sigma. This prevents tiny local noise
        # estimates from making normal pixels look like huge z-score outliers.
        "sigma_floor_percentile": 5.0,
        # Beamstop handling: "auto" keeps the existing detector, "off" disables it.
        "beamstop": "auto",
        # Low-intensity percentile used by the beamstop detector on the log image.
        "beamstop_low_percentile": 3.0,
        # Horizontal search range around the detected beamstop column.
        "beamstop_search_half_width": 120,
        # Maximum row-wise distance from the detected center column for accepting
        # a low-response run as part of the beamstop.
        "beamstop_max_anchor_distance": 50,
        # Accepted width range, in pixels, for low-response row runs in the
        # beamstop detector. These reject isolated speckles and huge regions.
        "beamstop_min_run_width": 3,
        "beamstop_max_run_width": 80,
        # Extra horizontal padding added to each accepted beamstop row/stripe.
        "beamstop_padding": 8,
        # Ellipse radii used to cover the beamstop tip around the first detected row.
        "beamstop_tip_radius_x": 24,
        "beamstop_tip_radius_y": 32,
        # Detector border width, in pixels, always marked bad. Set 0 to disable.
        "border": 10,
        # Optional final binary dilation iterations. Set 0 to leave the mask sharp.
        "dilate": 0,
    }

    def __init__(
        self,
        tiff_file: str | Path,
        *,
        baseline_mask_npy: str | Path | None = None,
        **parameters: Any,
    ) -> None:
        """Load the detector TIFF and optional starting mask.

        If ``baseline_mask_npy`` is omitted, the generator starts from an
        all-good scratch mask and only masks pixels found by later steps.
        """

        parameters = self._normalize_parameter_names(parameters)
        unknown = sorted(set(parameters) - set(self.DEFAULTS))
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}")

        self.tiff_file = Path(tiff_file).resolve()
        self.baseline_mask_npy = (
            Path(baseline_mask_npy).resolve() if baseline_mask_npy is not None else None
        )
        self.parameters = {**self.DEFAULTS, **parameters}

        self.raw = self.read_tiff(self.tiff_file, frame=int(self.parameters["frame"]))
        self.finite = np.isfinite(self.raw)
        if self.baseline_mask_npy is None:
            self.baseline_mask = np.zeros(self.raw.shape, dtype=bool)
        else:
            self.baseline_mask = self.load_baseline_mask(
                self.baseline_mask_npy,
                expected_shape=self.raw.shape,
            )

        self.log_normalized: np.ndarray | None = None
        self.local_median: np.ndarray | None = None
        self.local_residual: np.ndarray | None = None
        self.local_zscore: np.ndarray | None = None
        self.invalid_mask: np.ndarray | None = None
        self.dead_mask: np.ndarray | None = None
        self.hot_mask: np.ndarray | None = None
        self.beamstop_mask: np.ndarray | None = None
        self.final_mask: np.ndarray | None = None
        self.summary: dict[str, Any] = {}

    @classmethod
    def _normalize_parameter_names(cls, parameters: dict[str, Any]) -> dict[str, Any]:
        """Translate old parameter names to the current class API names."""

        normalized = dict(parameters)
        for old_name, new_name in cls.LEGACY_PARAMETER_ALIASES.items():
            if old_name not in normalized:
                continue
            if new_name in normalized:
                raise ValueError(
                    f"Use only one of {old_name!r} or {new_name!r}; {new_name!r} is preferred."
                )
            normalized[new_name] = normalized.pop(old_name)
        return normalized

    @staticmethod
    def find_first_tiff(folder: str | Path = ".") -> Path:
        """Return the first TIFF file in a folder for quick interactive use."""

        folder = Path(folder)
        candidates = sorted(
            path for path in folder.iterdir() if path.is_file() and path.suffix in TIFF_EXTENSIONS
        )
        if not candidates:
            raise FileNotFoundError(f"No TIFF files found in {folder}")
        return candidates[0]

    @staticmethod
    def read_tiff(path: str | Path, frame: int = 0) -> np.ndarray:
        """Read one 2D detector frame from a TIFF file as float32."""

        with Image.open(path) as image:
            if frame:
                image.seek(frame)
            array = np.asarray(image, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D detector image, got shape {array.shape}")
        return array

    @staticmethod
    def load_baseline_mask(path: str | Path, expected_shape: tuple[int, int]) -> np.ndarray:
        """Load a baseline .npy mask and convert nonzero values to bad pixels."""

        mask = np.load(path)
        if mask.shape != expected_shape:
            raise ValueError(
                f"Baseline mask shape {mask.shape} does not match detector image shape {expected_shape}"
            )
        return mask.astype(bool)

    @staticmethod
    def finite_percentile(array: np.ndarray, q: float) -> float:
        """Compute a percentile while ignoring NaN and infinite values."""

        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError("Image contains no finite pixels")
        return float(np.percentile(finite, q))

    def normalize_log_intensity(
        self,
        image: np.ndarray,
        low_percentile: float,
        high_percentile: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Shift, log-scale, and normalize intensities to a clipped 0-1 range."""

        finite = np.isfinite(image)
        low = self.finite_percentile(image, low_percentile)

        shifted = np.zeros_like(image, dtype=np.float32)
        shifted[finite] = np.maximum(image[finite] - low, 0.0)
        log_image = np.log1p(shifted, dtype=np.float32)

        high = self.finite_percentile(log_image, high_percentile)
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

    def median_filter_tiled(
        self,
        array: np.ndarray,
        window: int | None = None,
        tile_rows: int | None = None,
    ) -> np.ndarray:
        """Apply a median filter in row tiles to limit peak memory use.

        If ``window`` or ``tile_rows`` is omitted, the method uses the
        independent ``median_window`` and ``median_tile_rows`` class parameters.
        """

        window = int(self.parameters["median_window"] if window is None else window)
        tile_rows = int(self.parameters["median_tile_rows"] if tile_rows is None else tile_rows)

        if window % 2 != 1 or window < 3:
            raise ValueError("window must be an odd integer >= 3")

        pad = window // 2
        height, _width = array.shape
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
        self,
        analysis_image: np.ndarray,
        zscore_window: int,
        zscore_tile_rows: int,
        sigma_floor_percentile: float,
        use_median_filter: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute residual and robust z-score maps for bad-pixel candidates.

        With ``use_median_filter=True``, this uses tiled local medians and local
        MAD estimates. With ``use_median_filter=False``, it skips tiled median
        filtering and uses one global robust median/MAD estimate for the image.
        """

        if use_median_filter:
            local_median = self.median_filter_tiled(
                analysis_image,
                window=zscore_window,
                tile_rows=zscore_tile_rows,
            )
            residual = analysis_image - local_median

            abs_residual = np.abs(residual)
            local_mad = self.median_filter_tiled(
                abs_residual,
                window=zscore_window,
                tile_rows=zscore_tile_rows,
            )
            robust_sigma = (1.4826 * local_mad).astype(np.float32)

            positive_sigma = robust_sigma[np.isfinite(robust_sigma) & (robust_sigma > 0)]
            if positive_sigma.size:
                sigma_floor = float(np.percentile(positive_sigma, sigma_floor_percentile))
            else:
                sigma_floor = 1.0e-6
            sigma_floor = max(sigma_floor, 1.0e-6)
            zscore = residual / np.maximum(robust_sigma, sigma_floor)
        else:
            finite = np.isfinite(analysis_image)
            center = float(np.median(analysis_image[finite])) if finite.any() else 0.0
            local_median = np.full_like(analysis_image, center, dtype=np.float32)
            residual = analysis_image - local_median

            finite_residual = residual[np.isfinite(residual)]
            if finite_residual.size:
                global_mad = float(np.median(np.abs(finite_residual)))
                sigma_floor = max(1.4826 * global_mad, 1.0e-6)
            else:
                sigma_floor = 1.0e-6
            zscore = residual / sigma_floor

        return local_median, residual.astype(np.float32), zscore.astype(np.float32), sigma_floor

    @staticmethod
    def binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
        """Expand a boolean mask by one pixel per iteration in all directions."""

        result = mask.astype(bool, copy=True)
        for _ in range(iterations):
            padded = np.pad(result, 1, mode="constant", constant_values=False)
            expanded = np.zeros_like(result, dtype=bool)
            for dy in range(3):
                for dx in range(3):
                    expanded |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
            result = expanded
        return result

    def detect_beamstop_mask(
        self,
        analysis_image: np.ndarray,
        finite: np.ndarray,
        low_percentile: float,
        search_half_width: int,
        max_anchor_distance: int,
        min_run_width: int,
        max_run_width: int,
        padding: int,
        tip_radius_x: int,
        tip_radius_y: int,
        border: int,
    ) -> tuple[np.ndarray, dict[str, int | float | bool | None]]:
        """Build a candidate mask for a vertical low-response beamstop shadow."""

        height, width = analysis_image.shape
        mask = np.zeros_like(finite, dtype=bool)
        details: dict[str, int | float | bool | None] = {
            "beamstop_detected": False,
            "beamstop_center_col": None,
            "beamstop_top_row": None,
            "beamstop_bottom_row": None,
            "beamstop_low_threshold": None,
            "beamstop_candidate_rows": 0,
            "beamstop_stripe_half_width": None,
        }

        if not finite.any():
            return mask, details

        low_threshold = self.finite_percentile(analysis_image, low_percentile)
        low_mask = finite & (analysis_image <= low_threshold)

        lower_row0 = max(int(0.45 * height), border)
        lower_row1 = max(lower_row0, height - border)
        central_col0 = int(0.25 * width)
        central_col1 = int(0.75 * width)
        if lower_row1 <= lower_row0 or central_col1 <= central_col0:
            return mask, details

        column_scores = low_mask[lower_row0:lower_row1, central_col0:central_col1].sum(axis=0)
        if column_scores.size == 0:
            return mask, details

        smooth_window = min(15, column_scores.size)
        smoothed_scores = np.convolve(column_scores, np.ones(smooth_window), mode="same")
        center_col = central_col0 + int(np.argmax(smoothed_scores))
        peak_score = int(column_scores[center_col - central_col0])
        min_peak_score = max(25, int(0.05 * (lower_row1 - lower_row0)))
        if peak_score < min_peak_score:
            details.update(
                {
                    "beamstop_center_col": center_col,
                    "beamstop_low_threshold": low_threshold,
                }
            )
            return mask, details

        row0 = max(int(0.35 * height), border)
        row1 = max(row0, height - border)
        search_col0 = max(0, center_col - search_half_width)
        search_col1 = min(width, center_col + search_half_width + 1)
        row_extents: list[tuple[int, int, int, int]] = []

        for row in range(row0, row1):
            low_cols = np.flatnonzero(low_mask[row, search_col0:search_col1])
            if low_cols.size == 0:
                continue

            low_cols = low_cols + search_col0
            anchor_col = int(low_cols[np.argmin(np.abs(low_cols - center_col))])
            if abs(anchor_col - center_col) > max_anchor_distance:
                continue

            left = anchor_col
            right = anchor_col
            while left - 1 >= 0 and low_mask[row, left - 1]:
                left -= 1
            while right + 1 < width and low_mask[row, right + 1]:
                right += 1

            run_width = right - left + 1
            if min_run_width <= run_width <= max_run_width:
                mask[row, max(0, left - padding) : min(width, right + padding + 1)] = True
                row_extents.append((row, left, right, run_width))

        if not row_extents:
            details.update(
                {
                    "beamstop_center_col": center_col,
                    "beamstop_low_threshold": low_threshold,
                }
            )
            return mask, details

        top_row = min(extent[0] for extent in row_extents)
        bottom_row = max(extent[0] for extent in row_extents)
        run_widths = np.array([extent[3] for extent in row_extents], dtype=np.float32)
        stripe_half_width = int(np.ceil(np.percentile(run_widths, 95) / 2.0)) + padding
        mask[
            top_row : bottom_row + 1,
            max(0, center_col - stripe_half_width) : min(width, center_col + stripe_half_width + 1),
        ] = True
        if tip_radius_x > 0 and tip_radius_y > 0:
            yy, xx = np.ogrid[:height, :width]
            tip = ((xx - center_col) / tip_radius_x) ** 2 + ((yy - top_row) / tip_radius_y) ** 2 <= 1
            mask |= tip

        details.update(
            {
                "beamstop_detected": True,
                "beamstop_center_col": center_col,
                "beamstop_top_row": top_row,
                "beamstop_bottom_row": bottom_row,
                "beamstop_low_threshold": low_threshold,
                "beamstop_candidate_rows": len(row_extents),
                "beamstop_stripe_half_width": stripe_half_width,
            }
        )
        return mask, details

    def _merged_parameters(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Combine initialization defaults with per-call parameter overrides."""

        overrides = self._normalize_parameter_names(overrides)
        unknown = sorted(set(overrides) - set(self.DEFAULTS))
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}")
        return {**self.parameters, **overrides}

    def generate_mask(self, **parameter_overrides: Any) -> np.ndarray:
        """Generate and return the final uint8 mask without saving files.

        Example:
            generator = DetectorMaskGenerator(
                "image.tiff",
                baseline_mask_npy="baseline_mask.npy",
                border=10,
            )
            mask = generator.generate_mask(dead_z=-10, hot_z=15, beamstop="off")
        """

        params = self._merged_parameters(parameter_overrides)
        beamstop_mode = params["beamstop"]
        if beamstop_mode not in {"auto", "off"}:
            raise ValueError("beamstop must be 'auto' or 'off'")

        log_normalized, log_info = self.normalize_log_intensity(
            self.raw,
            low_percentile=float(params["log_low_percentile"]),
            high_percentile=float(params["log_high_percentile"]),
        )
        local_median, local_residual, local_zscore, sigma_floor = self.robust_local_zscore(
            log_normalized,
            zscore_window=int(params["zscore_window"]),
            zscore_tile_rows=int(params["zscore_tile_rows"]),
            sigma_floor_percentile=float(params["sigma_floor_percentile"]),
            use_median_filter=bool(params["use_median_filter"]),
        )

        invalid_mask = ~self.finite
        dead_mask = self.finite & (local_zscore <= float(params["dead_z"]))
        hot_mask = self.finite & (local_zscore >= float(params["hot_z"]))
        beamstop_mask = np.zeros_like(self.finite, dtype=bool)
        beamstop_info: dict[str, int | float | bool | None] = {
            "beamstop_detected": False,
            "beamstop_center_col": None,
            "beamstop_top_row": None,
            "beamstop_bottom_row": None,
            "beamstop_low_threshold": None,
            "beamstop_candidate_rows": 0,
            "beamstop_stripe_half_width": None,
        }

        if beamstop_mode == "auto":
            beamstop_mask, beamstop_info = self.detect_beamstop_mask(
                log_normalized,
                finite=self.finite,
                low_percentile=float(params["beamstop_low_percentile"]),
                search_half_width=int(params["beamstop_search_half_width"]),
                max_anchor_distance=int(params["beamstop_max_anchor_distance"]),
                min_run_width=int(params["beamstop_min_run_width"]),
                max_run_width=int(params["beamstop_max_run_width"]),
                padding=int(params["beamstop_padding"]),
                tip_radius_x=int(params["beamstop_tip_radius_x"]),
                tip_radius_y=int(params["beamstop_tip_radius_y"]),
                border=int(params["border"]),
            )

        final_mask_bool = self.baseline_mask | invalid_mask | dead_mask | hot_mask | beamstop_mask

        border = int(params["border"])
        if border > 0:
            final_mask_bool[:border, :] = True
            final_mask_bool[-border:, :] = True
            final_mask_bool[:, :border] = True
            final_mask_bool[:, -border:] = True

        dilate = int(params["dilate"])
        if dilate > 0:
            final_mask_bool = self.binary_dilate(final_mask_bool, dilate)

        self.log_normalized = log_normalized
        self.local_median = local_median
        self.local_residual = local_residual
        self.local_zscore = local_zscore
        self.invalid_mask = invalid_mask
        self.dead_mask = dead_mask
        self.hot_mask = hot_mask
        self.beamstop_mask = beamstop_mask
        self.final_mask = final_mask_bool.astype(np.uint8)

        finite_raw = self.raw[self.finite]
        bad_count = int(self.final_mask.sum())
        total_count = int(self.final_mask.size)
        self.summary = {
            "input_image": str(self.tiff_file),
            "baseline_mask_npy": str(self.baseline_mask_npy) if self.baseline_mask_npy else None,
            "shape": list(self.raw.shape),
            "dtype_after_read": str(self.raw.dtype),
            "raw_min": float(np.min(finite_raw)),
            "raw_max": float(np.max(finite_raw)),
            "raw_mean": float(np.mean(finite_raw)),
            "raw_median": float(np.median(finite_raw)),
            "log_low_percentile": float(params["log_low_percentile"]),
            "log_high_percentile": float(params["log_high_percentile"]),
            **log_info,
            "zscore_mode": "local_median_mad" if bool(params["use_median_filter"]) else "global_mad",
            "use_median_filter": bool(params["use_median_filter"]),
            "zscore_window": int(params["zscore_window"]),
            "zscore_tile_rows": int(params["zscore_tile_rows"]),
            "median_window": int(params["median_window"]),
            "median_tile_rows": int(params["median_tile_rows"]),
            "dead_z_threshold": float(params["dead_z"]),
            "hot_z_threshold": float(params["hot_z"]),
            "sigma_floor": sigma_floor,
            "dilate_iterations": dilate,
            "border_pixels_marked_bad": border,
            "baseline_mask_count": int(self.baseline_mask.sum()),
            "invalid_pixel_count": int(invalid_mask.sum()),
            "dead_candidate_count": int(dead_mask.sum()),
            "hot_candidate_count": int(hot_mask.sum()),
            "beamstop_mode": beamstop_mode,
            "beamstop_candidate_count": int(beamstop_mask.sum()),
            **beamstop_info,
            "bad_pixel_count": bad_count,
            "total_pixel_count": total_count,
            "bad_pixel_fraction": bad_count / total_count,
        }
        return self.final_mask

    def save_mask_npy(self, output_path: str | Path, mask: np.ndarray | None = None) -> Path:
        """Save a generated mask, or ``self.final_mask``, as a uint8 .npy file.

        Call ``generate_mask()`` first unless you pass an explicit ``mask``.
        The returned path is resolved so notebooks can report exactly what was
        written.
        """

        if mask is None:
            if self.final_mask is None:
                raise ValueError("No mask available. Call generate_mask() before save_mask_npy().")
            mask = self.final_mask

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mask.astype(np.uint8))
        return output_path

    def save_summary_json(self, output_path: str | Path) -> Path:
        """Write ``self.summary`` to a JSON file.

        Call ``generate_mask()`` first so the summary includes the current
        parameter values, component mask counts, and final mask statistics.
        """

        if not self.summary:
            raise ValueError("No summary available. Call generate_mask() before save_summary_json().")

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")
        return output_path
