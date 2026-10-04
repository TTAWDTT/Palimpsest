"""Test a bounded monotone *composite* transfer on held-out CSGC templates."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.special import expit


ARCHIVE = Path(r"E:\ai_image_origin_research\data\raw\csgc\CSGC_scan2400spi.zip")
OUT = Path("work/csgc_nonlinear_tone.json")
SIGMA = 2.5
INNER = np.s_[8:-8, 8:-8]


def read_pair(archive: zipfile.ZipFile, item: int) -> tuple[np.ndarray, np.ndarray]:
    code = f"Scan{item:05d}.tif"
    with Image.open(
        io.BytesIO(archive.read(f"2400dpi_NEW/bin_exact_crop_4/{code}"))
    ) as image:
        binary = np.asarray(image, dtype=np.float32) / 255
    with Image.open(
        io.BytesIO(archive.read(f"2400dpi_NEW/resize_exact_crop/{code}"))
    ) as image:
        scan = np.asarray(image, dtype=np.float32) / 255
    return gaussian_filter(binary, SIGMA, mode="reflect")[INNER], scan[INNER]


def curve(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    low, span, steepness, midpoint = p
    return low + span * expit(steepness * (x - midpoint))


with zipfile.ZipFile(ARCHIVE) as archive:
    rng = np.random.default_rng(20260924)
    xx = []
    yy = []
    for item in range(1, 101):
        source, real = read_pair(archive, item)
        chosen = rng.choice(source.size, size=2000, replace=False)
        xx.append(source.ravel()[chosen])
        yy.append(real.ravel()[chosen])
    x = np.concatenate(xx)
    y = np.concatenate(yy)
    fit = least_squares(
        lambda p: curve(x, p) - y,
        x0=np.array([0.0, 0.5, 15.0, 0.5]),
        bounds=([0, 0, 0.1, 0], [0.8, 1, 100, 1]),
        max_nfev=200,
    )
    if not fit.success:
        raise RuntimeError(f"nonlinear fit did not converge: {fit.message}")
    p = fit.x
    holdout = []
    for item in range(101, 951):
        source, real = read_pair(archive, item)
        predicted = curve(source, p)
        residual = predicted - real
        holdout.append(
            {
                "id": item,
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "pearson": float(np.corrcoef(predicted.ravel(), real.ravel())[0, 1]),
            }
        )


report = {
    "calibration_ids": "00001-00100",
    "holdout_ids": "00101-00950",
    "sigma_scan_pixels_fixed_from_prior_calibration": SIGMA,
    "curve": "low + span * sigmoid(steepness * (blurred_source - midpoint))",
    "fit_parameters": {
        name: float(value)
        for name, value in zip(("low", "span", "steepness", "midpoint"), p)
    },
    "calibration_sampled_pixels": int(len(x)),
    "calibration_sample_rmse": float(np.sqrt(np.mean((curve(x, p) - y) ** 2))),
    "holdout_quantiles": {
        metric: {
            str(q): float(np.quantile([r[metric] for r in holdout], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        }
        for metric in ("mae", "rmse", "pearson")
    },
    "limitation": "bounded monotone output diagnostic; not a measured scanner tone curve or a full print-physical simulator",
}
OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
