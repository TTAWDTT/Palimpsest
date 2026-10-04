"""Estimate a *composite* blur/linear-tone diagnostic on known CSGC pairs.

This is not a decomposition of printer deposition, paper scatter, and scanner
optics. The prepared public TIFFs have 72-DPI headers; physical 2400 SPI is
from the dataset's acquisition documentation. Templates 1..100 calibrate,
101..950 are never used in parameter selection.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import io
import json
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


ARCHIVE = DATA_ROOT / "raw/csgc/CSGC_scan2400spi.zip"
OUT = WORK_DIR / "csgc_effective_channel.json"
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
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
    return binary, scan


def fit_from_sums(sums: dict) -> dict:
    n, sx, sy, sxx, sxy, syy = (
        sums[key] for key in ("n", "sx", "sy", "sxx", "sxy", "syy")
    )
    mean_x, mean_y = sx / n, sy / n
    variance_x = sxx / n - mean_x**2
    covariance = sxy / n - mean_x * mean_y
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    mse = (syy / n - mean_y**2) - covariance**2 / variance_x
    return {
        "intercept": intercept,
        "slope": slope,
        "calibration_mse": mse,
        "calibration_rmse": float(np.sqrt(mse)),
    }


with zipfile.ZipFile(ARCHIVE) as archive:
    sums = {
        sigma: {k: 0.0 for k in ("n", "sx", "sy", "sxx", "sxy", "syy")}
        for sigma in SIGMAS
    }
    for item in range(1, 101):
        source, real = read_pair(archive, item)
        real = real[INNER]
        for sigma in SIGMAS:
            pred = (
                gaussian_filter(source, sigma=sigma, mode="reflect")
                if sigma
                else source
            )[INNER]
            stat = sums[sigma]
            stat["n"] += source.size
            stat["sx"] += float(pred.sum(dtype=np.float64))
            stat["sy"] += float(real.sum(dtype=np.float64))
            stat["sxx"] += float(np.sum(pred * pred, dtype=np.float64))
            stat["sxy"] += float(np.sum(pred * real, dtype=np.float64))
            stat["syy"] += float(np.sum(real * real, dtype=np.float64))
    fits = {sigma: fit_from_sums(stat) for sigma, stat in sums.items()}
    best = min(SIGMAS, key=lambda sigma: fits[sigma]["calibration_mse"])
    evaluated = {0.0: [], best: []}
    for item in range(101, 951):
        source, real = read_pair(archive, item)
        real = real[INNER]
        for sigma in evaluated:
            pred = (
                gaussian_filter(source, sigma=sigma, mode="reflect")
                if sigma
                else source
            )[INNER]
            fit = fits[sigma]
            pred = fit["intercept"] + fit["slope"] * pred
            residual = pred - real
            evaluated[sigma].append(
                {
                    "id": item,
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "pearson": float(np.corrcoef(pred.ravel(), real.ravel())[0, 1]),
                }
            )


def aggregate(rows: list[dict]) -> dict:
    return {
        feature: {
            str(q): float(np.quantile([r[feature] for r in rows], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        }
        for feature in ("mae", "rmse", "pearson")
    }


report = {
    "archive_sha256": json.loads(
        WORK_DIR / "csgc2400_full_audit.json".read_text(encoding="utf-8")
    )["archive_sha256"],
    "calibration_ids": "00001-00100",
    "holdout_ids": "00101-00950",
    "calibration_count": 100,
    "holdout_count": 850,
    "inner_roi_xyxy": [8, 8, 392, 392],
    "source": "exact 4x enlargement of 100x100 binary reference, prepared 400x400 TIFF",
    "model": "isotropic Gaussian blur on prepared binary reference plus shared linear intercept/slope",
    "limitation": "composite published-channel diagnostic; cannot split printer, paper, scanner, registration, or prepared-crop effects",
    "sigma_grid_scan_pixels": SIGMAS,
    "calibration_fits": {str(sigma): fit for sigma, fit in fits.items()},
    "selected_sigma_scan_pixels": best,
    "holdout_aggregates": {
        str(sigma): aggregate(rows) for sigma, rows in evaluated.items()
    },
    "holdout_per_template": {str(sigma): rows for sigma, rows in evaluated.items()},
}
OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            key: report[key]
            for key in (
                "calibration_count",
                "holdout_count",
                "selected_sigma_scan_pixels",
                "holdout_aggregates",
            )
        },
        indent=2,
    )
)
