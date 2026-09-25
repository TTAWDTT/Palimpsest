"""Evaluate a fixed fast display-prefilter numerical approximation on real ROIs.

This compares numerical output to the previously cached fine renderer and
its score on RAW. All physical parameters and count weights are unchanged.
"""

import csv
import json
from pathlib import Path
import time

import numpy as np

from origin_simulation.screen_capture import ScreenCaptureParameters, render_screen_capture
from work.audit_raw2event_split_first_frames import load_originals
from work.evaluate_raw2event_phase_confirmation import metrics, predict_phase
from work.evaluate_raw2event_spectral_mix import cached_bands, code_hash
from work.probe_raw2event_source_to_raw import prepare
from work.probe_screen_display_prefilter import approximate_bands, geometry_diagnostics


MANIFEST = Path("E:/ai_image_origin_research/data/manifests/raw2event_phase_confirmation_v1.csv")
GEOM = Path("work/raw2event_sensor_geometry_audit.json")
CORNERS = Path("work/raw2event_phase_confirmation_evaluation.json")
FINE_FIT = Path("work/raw2event_global_geometry_process_evaluation.json")
OUT = Path("work/raw2event_display_prefilter_evaluation.json")
DISPLAY_SAMPLES = 8
SENSOR_SAMPLES = 4


def centered_spectrum_deltas(fine_mosaic: np.ndarray, fast_mosaic: np.ndarray,
                             actual_mosaic: np.ndarray, mask: np.ndarray) -> dict:
    yy, xx = np.nonzero(mask)
    cy, cx = int(np.rint(yy.mean())), int(np.rint(xx.mean()))
    half = 32
    square = (slice(cy-half, cy+half), slice(cx-half, cx+half))
    if fine_mosaic[square].shape != (64, 64) or not np.all(mask[square]):
        raise RuntimeError("central 64x64 spectral crop not entirely inside display content")
    window = np.outer(np.hanning(64), np.hanning(64))
    def power(values: np.ndarray) -> np.ndarray:
        crop = values[square].astype(np.float64)
        return np.abs(np.fft.fft2((crop - crop.mean()) * window)) ** 2
    a, b, truth = power(fine_mosaic), power(fast_mosaic), power(actual_mosaic)
    fy, fx = np.meshgrid(np.fft.fftfreq(64), np.fft.fftfreq(64), indexing="ij")
    radius = np.hypot(fx, fy)
    bands = (("low_003_012", .03, .12), ("mid_012_025", .12, .25), ("high_025_05", .25, .5))
    return {name: {"delta_db": float(10 * np.log10((b[(radius >= lo) & (radius < hi)].sum() + 1e-12) /
                                                    (a[(radius >= lo) & (radius < hi)].sum() + 1e-12))),
                   "fine_power_fraction": float(a[(radius >= lo) & (radius < hi)].sum() / (a.sum() + 1e-12)),
                   "fast_power_fraction": float(b[(radius >= lo) & (radius < hi)].sum() / (b.sum() + 1e-12)),
                   "actual_power_fraction": float(truth[(radius >= lo) & (radius < hi)].sum() / (truth.sum() + 1e-12)),
                   "fine_to_actual_db": float(10 * np.log10((a[(radius >= lo) & (radius < hi)].sum() + 1e-12) /
                                                           (truth[(radius >= lo) & (radius < hi)].sum() + 1e-12))),
                   "fast_to_actual_db": float(10 * np.log10((b[(radius >= lo) & (radius < hi)].sum() + 1e-12) /
                                                           (truth[(radius >= lo) & (radius < hi)].sum() + 1e-12)))}
            for name, lo, hi in bands}


def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    if len(rows) != 10:
        raise RuntimeError("fixed ten already-inspected sources required")
    originals = load_originals(rows)
    corners = {r["prefix"]: r["geometry"]["source_rgb_refined_corners"]
               for r in json.loads(CORNERS.read_text(encoding="utf-8"))["per_source"]}
    affine = np.asarray(json.loads(GEOM.read_text(encoding="utf-8"))["fit_matrices"]["global_affine"])
    weights = np.asarray(json.loads(FINE_FIT.read_text(encoding="utf-8"))["methods"]["vertical_rgb"]
                         ["weights_intercept_gain_by_display_primary"])
    results = []
    for row in rows:
        prefix = row["prefix"]
        prepared = prepare(prefix, None, originals[prefix], np.asarray(corners[prefix], np.float32), affine)
        fine, fine_seconds = cached_bands(row, prepared, "vertical_rgb", code_hash())
        start = time.perf_counter()
        fast = render_screen_capture(
            prepared["drive"], prepared["actual"].shape,
            ScreenCaptureParameters(sensor_to_display=prepared["H"], fill_fraction=.85,
                                    emitter_layout="vertical_rgb", display_gamma=2.2,
                                    optical_blur_sigma_sensor_pixels=.8),
            spatial_method="prefilter").emitter_band_irradiance
        fast_seconds = time.perf_counter() - start
        if not results:
            independent = approximate_bands(prepared["drive"], prepared["actual"].shape,
                                            prepared["H"], .85, .8, "vertical_rgb",
                                            DISPLAY_SAMPLES, SENSOR_SAMPLES)
            if not np.array_equal(fast, independent):
                raise RuntimeError("integrated prefilter differs from independent prototype")
        select = prepared["mask"]
        diff = np.abs(fast[select] - fine[select])
        fine_mosaic = predict_phase(fine, "BGGR", weights)
        fast_mosaic = predict_phase(fast, "BGGR", weights)
        item = {"prefix": prefix, "class_name": row["class_name"],
                "geometry_diagnostics": geometry_diagnostics(prepared["H"], prepared["actual"].shape),
                "radiance_mean_abs_difference_to_fine": float(diff.mean()),
                "radiance_p99_abs_difference_to_fine": float(np.quantile(diff, .99)),
                "fine_seconds": fine_seconds, "fast_seconds": fast_seconds,
                "fine_raw": metrics(fine_mosaic, prepared),
                "fast_raw": metrics(fast_mosaic, prepared),
                "center64_mosaic_power_delta_db": centered_spectrum_deltas(
                    fine_mosaic, fast_mosaic, prepared["actual"], prepared["mask"])}
        results.append(item)
        print(f"fast comparison {len(results)}/10 {row['class_name']}", flush=True)
    report = {"scope": "ten already-inspected Raw2Event phase-confirmation sources; no parameter tuning on these RAW frames",
              "display_samples": DISPLAY_SAMPLES, "sensor_samples": SENSOR_SAMPLES,
              "fast_approximation": "integrated guarded prefilter: exact fractional display-raster cell coverage, center-Jacobian diagonal Gaussian, sensor-plane quadrature",
              "reference": "existing fine renderer at its own adaptive oversampling, not mathematical ground truth",
              "same_frozen_count_weights": True,
              "physical_limit": "LCD raster192 and optical sigma0.8 are virtual; center Jacobian and omitted covariance can fail for strong tilt",
              "per_source": results,
              "aggregate": {"mean_radiance_abs_difference": float(np.mean([r["radiance_mean_abs_difference_to_fine"] for r in results])),
                            "mean_fine_raw_mae_counts": float(np.mean([r["fine_raw"]["mae_counts"] for r in results])),
                            "mean_fast_raw_mae_counts": float(np.mean([r["fast_raw"]["mae_counts"] for r in results])),
                            "median_fine_seconds": float(np.median([r["fine_seconds"] for r in results])),
                            "median_fast_seconds": float(np.median([r["fast_seconds"] for r in results])),
                            "center64_power_delta_db_mean": {band: float(np.mean([r["center64_mosaic_power_delta_db"][band]["delta_db"] for r in results]))
                                                             for band in results[0]["center64_mosaic_power_delta_db"]}}}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
