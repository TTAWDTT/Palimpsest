"""Phase-only bispectrum diagonal; finite-image recapture invariance is not claimed."""

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import cv2
import numpy as np

from palimpsest.detection.algorithms.local_statistics.features import bounded_rgb

SIDE = 128
BANDS = ((2, 6), (6, 12), (12, 24))
FEATURE_NAMES = tuple(f"band{i}_{name}" for i in range(3) for name in ("cos1", "cos2", "cos4", "support"))
DECISION_INDICES = (1, 2, 5, 6, 9, 10)
RELATIVE_FLOOR = 1e-8


@lru_cache(maxsize=1)
def frequency_indices():
    k = np.arange(-SIDE // 2, SIDE // 2)
    y, x = np.meshgrid(k, k, indexing="ij")
    radius = np.hypot(y, x)
    return tuple((y[(radius >= lo) & (radius < hi)], x[(radius >= lo) & (radius < hi)])
                 for lo, hi in BANDS)


def patch_features(gray):
    gray = np.asarray(gray, dtype=float)
    if gray.shape != (SIDE, SIDE) or not np.isfinite(gray).all():
        raise ValueError("Expected finite 128x128 grayscale patch")
    transform = np.fft.fft2(gray - gray.mean())
    amplitude = np.abs(transform)
    floor = RELATIVE_FLOOR * amplitude.max()
    output = []
    for y, x in frequency_indices():
        first, second = transform[y, x], transform[2 * y, 2 * x]
        valid = (np.abs(first) > floor) & (np.abs(second) > floor)
        if not valid.any():
            output.extend((0.0, 0.0, 0.0, 0.0))
            continue
        u, v = first[valid] / np.abs(first[valid]), second[valid] / np.abs(second[valid])
        closure = u * u * np.conj(v)
        squared = closure * closure
        output.extend((float(closure.real.mean()), float(squared.real.mean()),
                       float((squared * squared).real.mean()), float(valid.mean())))
    return np.clip(output, -1, 1)


@dataclass(frozen=True)
class PhaseFeatures:
    values: np.ndarray
    patches: int
    preprocess_ms: float
    statistics_ms: float


def extract_features(image):
    start = perf_counter()
    gray = bounded_rgb(image, 512).astype(float) @ np.array([0.299, 0.587, 0.114])
    height, width = gray.shape
    if min(height, width) < SIDE:
        gray = cv2.copyMakeBorder(gray, 0, max(0, SIDE - height), 0, max(0, SIDE - width), cv2.BORDER_REFLECT_101)
        height, width = gray.shape
    positions = sorted({(y, x) for y in (0, height - SIDE) for x in (0, width - SIDE)})
    prepared = perf_counter()
    values = np.median([patch_features(gray[y:y + SIDE, x:x + SIDE]) for y, x in positions], axis=0)
    return PhaseFeatures(values, len(positions), (prepared - start) * 1000, (perf_counter() - prepared) * 1000)
