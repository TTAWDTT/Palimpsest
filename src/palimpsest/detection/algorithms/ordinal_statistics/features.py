"""Four-neighbor D4 orbit frequencies; invariance is scoped to gray input."""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from palimpsest.detection.algorithms.local_statistics.features import bounded_rgb

RADII = (1, 2, 4, 8)
ORBIT = np.array([0, 1, 1, 2, 1, 3, 2, 4, 1, 2, 3, 4, 2, 4, 4, 5], dtype=np.uint8)
FEATURE_NAMES = tuple(f"r{r}_{name}" for r in RADII
                      for name in ("zero", "one", "two_adjacent", "two_opposite", "three", "four", "ties"))


def gray_features(gray):
    """Gray-array operator, without resampling/quantization; four cardinal neighbors."""
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2 or not gray.size or not np.isfinite(gray).all():
        raise ValueError("Expected finite nonempty grayscale array")
    height, width = gray.shape
    pad = max(RADII)
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    values = []
    for radius in RADII:
        code = np.zeros(gray.shape, dtype=np.uint8)
        equal = 0
        for bit, (dy, dx) in enumerate(((-radius, 0), (0, radius), (radius, 0), (0, -radius))):
            neighbor = padded[pad + dy:pad + dy + height, pad + dx:pad + dx + width]
            code |= (neighbor >= gray).astype(np.uint8) << bit
            equal += np.count_nonzero(neighbor == gray)
        values.extend((np.bincount(ORBIT[code].ravel(), minlength=6) / gray.size).tolist())
        values.append(equal / (4 * gray.size))
    return np.asarray(values)


@dataclass(frozen=True)
class OrdinalFeatures:
    values: np.ndarray
    preprocess_ms: float
    statistics_ms: float


def extract_features(image):
    start = perf_counter()
    rgb = bounded_rgb(image, 512).astype(np.uint32)
    gray = 77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2]
    prepared = perf_counter()
    values = gray_features(gray)
    return OrdinalFeatures(values, (prepared - start) * 1000, (perf_counter() - prepared) * 1000)
