"""256-pixel ordinal and normalized RGB second-difference co-occurrences."""

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import cv2
import numpy as np

from palimpsest.contracts import validate_rgb
from palimpsest.detection.algorithms.ordinal_statistics.features import gray_features, FEATURE_NAMES as ORDINAL_NAMES

TRIPLES = tuple(product(range(-2, 3), repeat=3))


def canonical(triple):
    a = tuple(triple)
    return min(a, a[::-1], tuple(-x for x in a), tuple(-x for x in a[::-1]))


ORBITS = tuple(sorted({canonical(t) for t in TRIPLES}))
LOOKUP = np.array([ORBITS.index(canonical(t)) for t in TRIPLES], dtype=np.uint8)
RESIDUAL_NAMES = tuple(f'rgb_{channel}_scale{scale}_orbit{index}' for channel in 'rgb'
                       for scale in (1, 2, 4) for index in range(len(ORBITS)))
FEATURE_NAMES = tuple('ordinal_' + n for n in ORDINAL_NAMES) + RESIDUAL_NAMES


def resize256(image):
    validate_rgb(image)
    height, width = image.shape[:2]
    ratio = min(1.0, 256 / max(height, width))
    if ratio == 1:
        return image
    return cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio))),
                      interpolation=cv2.INTER_AREA)


def quantize(residual):
    residual = np.asarray(residual, dtype=np.float32)
    if residual.ndim != 2 or not residual.size or not np.isfinite(residual).all():
        raise ValueError("Expected finite residual plane")
    local = cv2.boxFilter(np.abs(residual), -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(np.rint(2 * residual / (local + 1)), -2, 2).astype(np.int8)


def triplet_counts(quantized, axis):
    q = np.asarray(quantized)
    if (q.ndim != 2 or min(q.shape) < 3 or axis not in (0, 1) or q.dtype.kind not in 'iu'
            or np.any(q < -2) or np.any(q > 2)):
        raise ValueError("Expected integer quantized plane in [-2,2]")
    q = q.astype(np.int16) + 2
    a, b, c = (q[:-2], q[1:-1], q[2:]) if axis == 0 else (q[:, :-2], q[:, 1:-1], q[:, 2:])
    codes = a * 25 + b * 5 + c
    return np.bincount(LOOKUP[codes].ravel(), minlength=len(ORBITS))


def residual_histogram(channel):
    channel = np.asarray(channel, dtype=np.float32)
    if channel.ndim != 2 or min(channel.shape) < 5 or not np.isfinite(channel).all():
        raise ValueError("Expected finite residual input with side >=5")
    middle = channel[1:-1, 1:-1]
    horizontal = channel[1:-1, :-2] + channel[1:-1, 2:] - 2 * middle
    vertical = channel[:-2, 1:-1] + channel[2:, 1:-1] - 2 * middle
    counts = triplet_counts(quantize(horizontal), 1) + triplet_counts(quantize(vertical), 0)
    return counts / counts.sum()


@dataclass(frozen=True)
class ResidualFeatures:
    values: np.ndarray
    preprocess_ms: float
    statistics_ms: float


def extract_features(image, *, residual=True):
    start = perf_counter()
    rgb = resize256(image)
    if min(rgb.shape[:2]) < 20:
        bottom, right = max(0, 20-rgb.shape[0]), max(0, 20-rgb.shape[1])
        rgb = cv2.copyMakeBorder(rgb, 0, bottom, 0, right, cv2.BORDER_REFLECT_101)
    prepared = perf_counter()
    integer = rgb.astype(np.uint32)
    gray = 77 * integer[..., 0] + 150 * integer[..., 1] + 29 * integer[..., 2]
    ordinal = gray_features(gray)
    if not residual:
        return ResidualFeatures(ordinal, (prepared-start)*1000, (perf_counter()-prepared)*1000)
    pyramids = [rgb.astype(np.float32)]
    for factor in (2, 4):
        pyramids.append(cv2.resize(rgb.astype(np.float32), (max(5, rgb.shape[1]//factor), max(5, rgb.shape[0]//factor)),
                                   interpolation=cv2.INTER_AREA))
    residual = np.concatenate([residual_histogram(p[..., channel]) for channel in range(3) for p in pyramids])
    return ResidualFeatures(np.r_[ordinal, residual], (prepared-start)*1000, (perf_counter()-prepared)*1000)
