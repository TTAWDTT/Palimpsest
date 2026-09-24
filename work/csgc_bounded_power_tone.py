"""Diagnostic: can one bounded reflectance-plus-power curve explain CSGC?"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares


ARCHIVE = Path(r"E:\ai_image_origin_research\data\raw\csgc\CSGC_scan2400spi.zip")
OUT = Path("work/csgc_bounded_power_tone.json")
INNER = np.s_[8:-8, 8:-8]
SIGMA = 2.5


def pair(z: zipfile.ZipFile, item: int) -> tuple[np.ndarray, np.ndarray]:
    name = f"Scan{item:05d}.tif"
    with Image.open(io.BytesIO(z.read(f"2400dpi_NEW/bin_exact_crop_4/{name}"))) as im:
        x = np.asarray(im, dtype=np.float32) / 255
    with Image.open(io.BytesIO(z.read(f"2400dpi_NEW/resize_exact_crop/{name}"))) as im:
        y = np.asarray(im, dtype=np.float32) / 255
    return gaussian_filter(x, SIGMA, mode="reflect")[INNER], y[INNER]


def response(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    ink_floor, paper_span_fraction, exponent = p
    paper_minus_ink = (1 - ink_floor) * paper_span_fraction
    return np.power(ink_floor + paper_minus_ink * x, exponent)


with zipfile.ZipFile(ARCHIVE) as z:
    rng = np.random.default_rng(20260924)
    xx, yy = [], []
    for item in range(1, 101):
        x, y = pair(z, item)
        chosen = rng.choice(x.size, size=2000, replace=False)
        xx.append(x.ravel()[chosen])
        yy.append(y.ravel()[chosen])
    train_x, train_y = np.concatenate(xx), np.concatenate(yy)
    fit = least_squares(lambda p: response(train_x, p) - train_y,
                        x0=[.02, .85, 2.0], bounds=([0, 0, .1], [.6, 1, 8]),
                        max_nfev=250)
    if not fit.success:
        raise RuntimeError(fit.message)
    p = fit.x
    scores = []
    for item in range(101, 951):
        x, y = pair(z, item)
        pred = response(x, p)
        scores.append({"id": item, "mae": float(np.mean(np.abs(pred - y))),
                       "rmse": float(np.sqrt(np.mean((pred - y)**2))),
                       "pearson": float(np.corrcoef(pred.ravel(), y.ravel())[0, 1])})

report = {
    "model": "(ink_floor + (1-ink_floor)*paper_span_fraction*GaussianBlur(source,2.5px)) ** exponent",
    "parameter_status": "bounded composite diagnostic, not individually measured material or ISP values",
    "calibration_ids": "00001-00100", "holdout_ids": "00101-00950",
    "parameters": {name: float(value) for name, value in zip(
        ("ink_floor", "paper_span_fraction", "exponent"), p)},
    "implied_paper_level_before_power": float(p[0] + (1 - p[0]) * p[1]),
    "calibration_sample_rmse": float(np.sqrt(np.mean((response(train_x, p) - train_y)**2))),
    "holdout_quantiles": {metric: {str(q): float(np.quantile([r[metric] for r in scores], q))
                                  for q in (.05, .25, .5, .75, .95)}
                          for metric in ("mae", "rmse", "pearson")},
}
OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
