"""Compare a printed uniform tile with blank paper in one DFD scan.

This is an observational control for the halftone peak, not a measurement of
printer RIP settings. Coordinates refer to the first verified P3 TIFF only.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE = Path(
    r"E:\ai_image_origin_research\data\derived\dfd_probe\D5_DC1_x_800_P3_S1_T1_2111_1.tiff"
)
ROIS = {
    "light_gray_tile": (5100, 1800, 5612, 2312),
    "medium_gray_tile": (3200, 1800, 3712, 2312),
    "dark_gray_tile": (1000, 1800, 1512, 2312),
    "blank_paper": (5650, 7050, 6162, 7562),
}


def probe(patch: np.ndarray) -> dict:
    window = np.hanning(512)
    spectrum = np.abs(
        np.fft.fftshift(
            np.fft.fft2((patch - patch.mean()) * window[:, None] * window[None, :])
        )
    )
    yy, xx = np.indices((512, 512))
    radius = np.hypot(xx - 256, yy - 256)
    annulus = (radius > 65) & (radius < 110)
    neighborhood = (np.abs(xx - 192) <= 1) & (np.abs(yy - 192) <= 1)
    peak = float(spectrum[neighborhood].max())
    return {
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "fixed_45deg_peak": peak,
        "annulus_median": float(np.median(spectrum[annulus])),
        "peak_over_annulus_median": peak
        / max(float(np.median(spectrum[annulus])), 1e-12),
    }


with Image.open(SOURCE) as image:
    results = {
        name: {"roi_xyxy": roi, **probe(np.asarray(image.crop(roi), dtype=np.float64))}
        for name, roi in ROIS.items()
    }
print(json.dumps(results, indent=2))
