"""Test chart-only projector forward models on held-out textured inputs.

This is a deliberately restricted, effective RGB experiment on the author's
warped 256px CompenNet PNGs. It does not recover physical projector, camera or
surface parameters individually. No test image is used to fit parameters.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import io
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


ARCHIVE = DATA_ROOT / "raw/compennet/CompenNetDataset.zip"
AUDIT = WORK_DIR / "compennet_full_audit.json"
OUT = WORK_DIR / "projector_chart_probe.json"
LEVELS = np.array([0, 64, 128, 191, 255], dtype=np.float32) / 255


def load_rgb(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with Image.open(io.BytesIO(archive.read(name))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def trilinear(cube: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Interpolate a (5,5,5,3) chart LUT at each source RGB pixel."""
    low = np.searchsorted(LEVELS, rgb, side="right") - 1
    low = np.clip(low, 0, 3)
    high = low + 1
    t = (rgb - LEVELS[low]) / (LEVELS[high] - LEVELS[low])
    result = np.zeros_like(rgb)
    for ri in (0, 1):
        wr = t[..., 0] if ri else 1 - t[..., 0]
        ii = high[..., 0] if ri else low[..., 0]
        for gi in (0, 1):
            wg = t[..., 1] if gi else 1 - t[..., 1]
            jj = high[..., 1] if gi else low[..., 1]
            for bi in (0, 1):
                wb = t[..., 2] if bi else 1 - t[..., 2]
                kk = high[..., 2] if bi else low[..., 2]
                result += cube[ii, jj, kk] * (wr * wg * wb)[..., None]
    return result


def read_chart(archive: zipfile.ZipFile, setup: str) -> tuple[np.ndarray, np.ndarray]:
    # Numeric chart sequence increments red first, then green, then blue.
    source = np.empty((125, 3), dtype=np.float32)
    captured = np.empty((125, 256, 256, 3), dtype=np.float32)
    for i in range(125):
        input_image = load_rgb(archive, f"ref/img_{i + 1:04d}.png")
        if not np.all(input_image == input_image[0, 0]):
            raise ValueError(f"reference {i + 1} is not a spatially uniform input")
        source[i] = input_image[0, 0]
        captured[i] = load_rgb(archive, f"{setup}/cam/warp/ref/img_{i + 1:04d}.png")
    expected = np.array(
        [(r, g, b) for b in LEVELS for g in LEVELS for r in LEVELS], dtype=np.float32
    )
    if not np.allclose(source, expected, atol=1 / 255):
        raise ValueError("numeric chart ordering differs from assumed RGB 5^3")
    return source, captured


def fit_factorized(chart: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Effective camera-domain black floor + surface gain * shared 3D LUT."""
    black = chart[0].copy()
    gain = chart[-1] - black
    usable = gain > 0.08
    normalized = np.where(
        usable[None], (chart - black[None]) / np.maximum(gain[None], 0.08), np.nan
    )
    # Medians remove areas with small dynamic range; not a spectral estimate.
    lut = np.nanmedian(normalized.reshape(125, -1, 3), axis=1).reshape(5, 5, 5, 3)
    lut = lut.transpose(2, 1, 0, 3).copy()  # physical RGB axis order
    lut[0, 0, 0] = 0
    lut[-1, -1, -1] = 1
    return black, gain, lut.astype(np.float32)


def fit_pixel_affine(source: np.ndarray, chart: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((125, 1), np.float32), source], axis=1)
    return (np.linalg.pinv(design) @ chart.reshape(125, -1)).reshape(4, 256, 256, 3)


def simulate(model: str, source: np.ndarray, params: tuple, sigma: float) -> np.ndarray:
    if sigma > 0:
        source = gaussian_filter(source, (sigma, sigma, 0), mode="reflect")
    if model == "factorized":
        black, gain, lut = params
        prediction = black + gain * trilinear(lut, source)
    elif model == "affine":
        (weights,) = params
        prediction = weights[0] + np.einsum("hwc,chwk->hwk", source, weights[1:])
    else:
        raise ValueError(model)
    return np.clip(prediction, 0, 1)


def score(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = prediction - target
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return {"mae": mae, "rmse": rmse, "psnr": float(-20 * np.log10(max(rmse, 1e-12)))}


def run_setup(
    archive: zipfile.ZipFile, setup: str, train_count: int, test_count: int
) -> dict:
    source_chart, captured_chart = read_chart(archive, setup)
    black, gain, lut = fit_factorized(captured_chart)
    affine = fit_pixel_affine(source_chart, captured_chart)
    chart_metrics = {}
    for model, params in (("factorized", (black, gain, lut)), ("affine", (affine,))):
        sampled = []
        for i in (0, 2, 12, 31, 62, 93, 124):
            prediction = simulate(
                model,
                source_chart[i][None, None].repeat(256, 0).repeat(256, 1),
                params,
                0,
            )
            sampled.append(score(prediction, captured_chart[i])["mae"])
        chart_metrics[model] = float(np.mean(sampled))
    sigmas = (0.0, 0.45, 0.9, 1.35)
    train = {
        model: {str(sigma): [] for sigma in sigmas}
        for model in ("factorized", "affine")
    }
    for index in range(1, train_count + 1):
        x = load_rgb(archive, f"train/img_{index:04d}.png")
        y = load_rgb(archive, f"{setup}/cam/warp/train/img_{index:04d}.png")
        for model, params in (
            ("factorized", (black, gain, lut)),
            ("affine", (affine,)),
        ):
            for sigma in sigmas:
                train[model][str(sigma)].append(
                    score(simulate(model, x, params, sigma), y)["mae"]
                )
    chosen = {
        model: min(sigmas, key=lambda sigma: np.mean(train[model][str(sigma)]))
        for model in train
    }
    test = {
        "input_only": [],
        "factorized_zero": [],
        "factorized_tuned": [],
        "affine_zero": [],
        "affine_tuned": [],
    }
    for index in range(1, test_count + 1):
        x = load_rgb(archive, f"test/img_{index:04d}.png")
        y = load_rgb(archive, f"{setup}/cam/warp/test/img_{index:04d}.png")
        test["input_only"].append(score(x, y))
        for model, params in (
            ("factorized", (black, gain, lut)),
            ("affine", (affine,)),
        ):
            test[f"{model}_zero"].append(score(simulate(model, x, params, 0), y))
            test[f"{model}_tuned"].append(
                score(simulate(model, x, params, chosen[model]), y)
            )
    aggregate = {
        key: {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in ("mae", "rmse", "psnr")
        }
        for key, values in test.items()
    }
    return {
        "setup": setup,
        "train_sigma_calibration_images": train_count,
        "test_images": test_count,
        "chosen_sigma": chosen,
        "train_mae_by_sigma": {
            m: {s: float(np.mean(v)) for s, v in vals.items()}
            for m, vals in train.items()
        },
        "chart_seven_color_mae": chart_metrics,
        "test_aggregate": aggregate,
        "test_by_image": test,
        "dynamic_range_usable_fraction": float(np.mean(gain > 0.08)),
        "lut_minmax": [float(lut.min()), float(lut.max())],
    }
