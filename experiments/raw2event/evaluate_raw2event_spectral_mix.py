"""Fit effective display-primary to RAW-CFA coupling on frozen calibration sources.

The source-to-screen and spatial assumptions remain virtual. The learned
nonnegative 3x3 coefficients combine unknown display spectra, optics and sensor
responses; they must not be described as an identified sensor spectrum.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
from palimpsest.io.provenance import simulation_code_files

import numpy as np
from scipy.optimize import nnls

from palimpsest.simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)
from experiments.raw2event.audit_raw2event_split_first_frames import load_originals
from experiments.raw2event.probe_raw2event_source_to_raw import (
    BLUR_SIGMA_SENSOR_PIXELS,
    prepare,
)


SPLIT = DATA_ROOT / "manifests/raw2event_process_split_v1.csv"
SPLIT_SHA = "471fbff8020f88c1b73664e5794285df28da4792e5fb77bdfe682b5ee9bb7a43"
GEOMETRY = WORK_DIR / "raw2event_content_registered_geometry.json"
CACHE = DATA_ROOT / "derived/raw2event_spectral_mix_v1"
OUT = WORK_DIR / "raw2event_spectral_mix_v1.json"
METHODS = ("vertical_rgb", "co_spatial_rgb_control", "simple_rgb_sample_control")
CODE = (
    *simulation_code_files(),
    Path("origin_simulation/screen_pipeline.py"),
    Path("experiments/raw2event/probe_raw2event_source_to_raw.py"),
    Path("experiments/raw2event/evaluate_raw2event_spectral_mix.py"),
)


def code_hash() -> str:
    digest = hashlib.sha256(SPLIT_SHA.encode())
    for path in CODE:
        digest.update(path.as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def render_bands(prepared: dict, method: str) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    if method == "simple_rgb_sample_control":
        height, width = prepared["actual"].shape
        yy, xx = np.indices((height, width), dtype=np.float32)
        xy = np.stack((xx + 0.5, yy + 0.5), axis=-1).reshape(1, -1, 2)
        uv = cv2.perspectiveTransform(xy, prepared["H"]).reshape(height, width, 2)
        radiance = prepared["drive"] ** 2.2
        mapped = cv2.remap(
            radiance,
            uv[..., 0].astype(np.float32),
            uv[..., 1].astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        bands = cv2.GaussianBlur(mapped, (0, 0), BLUR_SIGMA_SENSOR_PIXELS)
    else:
        params = ScreenCaptureParameters(
            sensor_to_display=prepared["H"],
            fill_fraction=0.85,
            emitter_layout=method,
            display_gamma=2.2,
            optical_blur_sigma_sensor_pixels=BLUR_SIGMA_SENSOR_PIXELS,
            exposure_electrons_per_unit=None,
            read_noise_electrons=0.0,
        )
        result = render_screen_capture(
            prepared["drive"],
            prepared["actual"].shape,
            params,
            spatial_method="fine",
            tile_size_sensor_pixels=96,
        )
        bands = result.emitter_band_irradiance
    return bands.astype(np.float32), time.perf_counter() - start


def cached_bands(
    row: dict, prepared: dict, method: str, fingerprint: str
) -> tuple[np.ndarray, float]:
    path = CACHE / fingerprint[:16] / method / f"{row['prefix']}.npz"
    key = hashlib.sha256(
        fingerprint.encode()
        + method.encode()
        + prepared["drive"].tobytes()
        + prepared["H"].tobytes()
        + np.asarray(prepared["roi"], dtype=np.int64).tobytes()
    ).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if data["key"].item() != key or data["bands"].shape != (
                *prepared["actual"].shape,
                3,
            ):
                raise RuntimeError(f"stale bands cache: {path}")
            return data["bands"].copy(), float(data["seconds"])
    bands, seconds = render_bands(prepared, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(
        partial, key=np.asarray(key), bands=bands, seconds=np.asarray(seconds)
    )
    partial.replace(path)
    return bands, seconds


def channel_mask(shape: tuple[int, int], channel: int) -> np.ndarray:
    y, x = np.indices(shape)
    if channel == 0:
        return (y % 2 == 0) & (x % 2 == 0)
    if channel == 2:
        return (y % 2 == 1) & (x % 2 == 1)
    return (y + x) % 2 == 1


def fit_nnls(
    entries: list[tuple[dict, dict, np.ndarray, float]], diagonal: bool
) -> np.ndarray:
    weights = np.zeros((3, 4), dtype=np.float64)
    for channel in range(3):
        gram = np.zeros((4, 4), dtype=np.float64)
        target = np.zeros(4, dtype=np.float64)
        for row, prepared, bands, _ in entries:
            if row["role"] != "calibration":
                continue
            select = prepared["mask"] & channel_mask(prepared["actual"].shape, channel)
            features = np.concatenate(
                (
                    np.ones((int(select.sum()), 1), dtype=np.float64),
                    bands[select].astype(np.float64),
                ),
                axis=1,
            )
            actual = prepared["actual"][select].astype(np.float64)
            gram += features.T @ features
            target += features.T @ actual
        columns = [0, channel + 1] if diagonal else [0, 1, 2, 3]
        reduced_gram = gram[np.ix_(columns, columns)]
        reduced_target = target[columns]
        lower = np.linalg.cholesky(reduced_gram)
        upper = lower.T
        reduced_weights, _ = nnls(upper, np.linalg.solve(lower, reduced_target))
        weights[channel, columns] = reduced_weights
    return weights


def evaluate(
    entries: list[tuple[dict, dict, np.ndarray, float]], weights: np.ndarray
) -> dict:
    records = []
    for row, prepared, bands, seconds in entries:
        pred = np.empty(prepared["actual"].shape, dtype=np.float32)
        for channel in range(3):
            mask = channel_mask(pred.shape, channel)
            pred[mask] = weights[channel, 0] + bands[mask] @ weights[channel, 1:]
        select = prepared["mask"]
        actual = prepared["actual"][select]
        predicted = pred[select]
        records.append(
            {
                "prefix": row["prefix"],
                "role": row["role"],
                "class_name": row["class_name"],
                "capture_day": row["capture_day"],
                "n": int(select.sum()),
                "mae_counts": float(np.abs(predicted - actual).mean()),
                "rmse_counts": float(np.sqrt(np.mean((predicted - actual) ** 2))),
                "pearson": float(np.corrcoef(predicted, actual)[0, 1]),
                "actual_mean": float(actual.mean()),
                "predicted_mean": float(predicted.mean()),
                "render_seconds": seconds,
            }
        )
    aggregate = {}
    for role in ("calibration", "development"):
        subset = [record for record in records if record["role"] == role]
        aggregate[role] = {
            "n_sources": len(subset),
            "mean_source_mae_counts": float(np.mean([r["mae_counts"] for r in subset])),
            "median_source_mae_counts": float(
                np.median([r["mae_counts"] for r in subset])
            ),
            "mean_source_pearson": float(np.mean([r["pearson"] for r in subset])),
            "pooled_pixel_mae_counts": float(
                sum(r["mae_counts"] * r["n"] for r in subset)
                / sum(r["n"] for r in subset)
            ),
        }
    return {
        "weights_intercept_and_display_primary_coupling": weights.tolist(),
        "median_render_seconds": float(
            np.median([r["render_seconds"] for r in records])
        ),
        **aggregate,
        "per_source": records,
    }


def main() -> None:
    if hashlib.sha256(SPLIT.read_bytes()).hexdigest() != SPLIT_SHA:
        raise RuntimeError("frozen split changed")
    rows = [
        row
        for row in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
        if row["role"] in ("calibration", "development")
    ]
    geometry = {
        row["prefix"]: row
        for row in json.loads(GEOMETRY.read_text(encoding="utf-8"))["records"]
    }
    if len(rows) != 20 or set(geometry) != {row["prefix"] for row in rows}:
        raise RuntimeError("complete twenty-source RGB-only registration required")
    originals = load_originals(rows)
    fingerprint = code_hash()
    entries = {method: [] for method in METHODS}
    for number, row in enumerate(rows, start=1):
        corners = np.asarray(
            geometry[row["prefix"]]["refined_corners"], dtype=np.float32
        )
        prepared = prepare(row["prefix"], None, originals[row["prefix"]], corners)
        for method in METHODS:
            bands, seconds = cached_bands(row, prepared, method, fingerprint)
            entries[method].append((row, prepared, bands, seconds))
        print(
            f"rendered bands {number}/20 {row['role']} {row['class_name']}", flush=True
        )
    conditions = {}
    for method in METHODS:
        for coupling in ("diagonal", "nonnegative_full"):
            weights = fit_nnls(entries[method], diagonal=(coupling == "diagonal"))
            conditions[f"{method}__{coupling}"] = evaluate(entries[method], weights)
    report = {
        "split_sha256": SPLIT_SHA,
        "code_fingerprint": fingerprint,
        "n_sources": 20,
        "n_calibration": 10,
        "n_development": 10,
        "geometry": "known-source RGB-only local optimization, selected after initial development RAW results",
        "fit": "nonnegative least squares, separate intercept per CFA color; calibration only",
        "coupling_interpretation": "effective display-primary-to-RAW response; cannot identify physical sensor spectral sensitivity separately",
        "conditions": conditions,
        "qualification": "exploratory post-hoc physical diagnosis, not a preregistered heldout validation; display raster, gamma, CFA phase and blur are virtual, reserved untouched",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value["development"] for key, value in conditions.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
