"""Fit effective display-primary to RAW-CFA coupling on frozen calibration sources.

The source-to-screen and spatial assumptions remain virtual. The learned
nonnegative 3x3 coefficients combine unknown display spectra, optics and sensor
responses; they must not be described as an identified sensor spectrum.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import hashlib
from pathlib import Path
import time

import cv2
from palimpsest.io.provenance import MODEL_ROOT, simulation_code_files

import numpy as np
from scipy.optimize import nnls

from palimpsest.simulation.screen_capture.capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)
from experiments.screen_capture.source_to_raw.raw2event.protocol import (
    BLUR_SIGMA_SENSOR_PIXELS,
)


SPLIT = DATA_ROOT / "manifests/raw2event_process_split_v1.csv"
SPLIT_SHA = "471fbff8020f88c1b73664e5794285df28da4792e5fb77bdfe682b5ee9bb7a43"
GEOMETRY = WORK_DIR / "raw2event_content_registered_geometry.json"
CACHE = DATA_ROOT / "derived/raw2event_spectral_mix_v1"
OUT = WORK_DIR / "raw2event_spectral_mix_v1.json"
METHODS = ("vertical_rgb", "co_spatial_rgb_control", "simple_rgb_sample_control")
CODE = (
    Path(__file__),
    *simulation_code_files(),
    MODEL_ROOT.parent / "data/video.py",
    MODEL_ROOT.parent / "data/raw2event.py",
    MODEL_ROOT.parent / "data/cifar10.py",
    Path("experiments/screen_capture/source_to_raw/raw2event/protocol.py"),
    Path("experiments/screen_capture/source_to_raw/raw2event/run_two_sources.py"),
    Path(
        "experiments/screen_capture/spectral_response/raw2event/evaluate_spectral_mix.py"
    ),
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
