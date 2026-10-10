"""Bounded spatial statistics, not a BRISQUE implementation or AI guarantee."""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from palimpsest.contracts import validate_rgb

FEATURE_NAMES = (
    "local_abs", "local_variance", "local_kurtosis",
    "local_corr_h", "local_corr_v", "local_corr_d1", "local_corr_d2",
    "scale_gradient_log_ratio", "scale_gradient_correlation",
    "scale_abs_change", "scale_variance_log_ratio", "scale_kurtosis_change",
    "color_corr_rg", "color_corr_rb", "color_corr_gb",
    "color_chroma_luma_gradient_log_ratio", "color_chroma_gradient_correlation",
)
FAMILIES = {name: tuple(i for i, feature in enumerate(FEATURE_NAMES)
                       if feature.startswith(name + "_")) for name in ("local", "scale", "color")}
EPS = 1e-8


@dataclass(frozen=True)
class FeatureResult:
    values: np.ndarray
    patches: int
    preprocess_ms: float
    statistics_ms: float


def bounded_rgb(image, long_edge):
    validate_rgb(image)
    if long_edge not in (512, 1024):
        raise ValueError("Registered long edges are 512 and 1024")
    height, width = image.shape[:2]
    ratio = min(1.0, long_edge / max(height, width))
    if ratio == 1:
        return image
    return cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio))),
                      interpolation=cv2.INTER_AREA)


def normalized_residual(channel):
    if np.ptp(channel) < EPS:
        return np.zeros_like(channel)
    mean = cv2.GaussianBlur(channel, (7, 7), 7 / 6, borderType=cv2.BORDER_REFLECT_101)
    second = cv2.GaussianBlur(channel * channel, (7, 7), 7 / 6,
                              borderType=cv2.BORDER_REFLECT_101)
    return (channel - mean) / (np.sqrt(np.maximum(second - mean * mean, 0)) + 1 / 255)


def cosine(left, right):
    denominator = np.sqrt(np.mean(left * left) * np.mean(right * right))
    return float(np.mean(left * right) / (denominator + EPS))


def moments(residual):
    variance = float(np.mean(residual * residual))
    return (float(np.mean(np.abs(residual))), variance,
            float(np.mean(residual ** 4) / (variance * variance + EPS)))


def gradient(channel):
    dx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8,
                   borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8,
                   borderType=cv2.BORDER_REFLECT_101)
    return np.hypot(dx, dy)


def patch_statistics(patch, family=None):
    rgb = patch.astype(np.float64) / 255
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    values = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    if family in (None, "local", "scale"):
        residual = normalized_residual(gray)
        first = moments(residual)
        if family in (None, "local"):
            values[:3] = first
            if min(gray.shape) > 1:
                values[3:7] = [
                    cosine(residual[:, :-1], residual[:, 1:]),
                    cosine(residual[:-1], residual[1:]),
                    cosine(residual[:-1, :-1], residual[1:, 1:]),
                    cosine(residual[:-1, 1:], residual[1:, :-1]),
                ]
        if family in (None, "scale"):
            small = cv2.resize(gray, (max(1, gray.shape[1] // 2), max(1, gray.shape[0] // 2)),
                               interpolation=cv2.INTER_AREA)
            second = moments(normalized_residual(small))
            coarse_gradient = gradient(small)
            fine_gradient = cv2.resize(gradient(gray), (small.shape[1], small.shape[0]),
                                       interpolation=cv2.INTER_AREA)
            values[7:12] = [
                np.log((np.mean(fine_gradient) + EPS) / (np.mean(coarse_gradient) + EPS)),
                cosine(fine_gradient - fine_gradient.mean(),
                       coarse_gradient - coarse_gradient.mean()),
                second[0] - first[0], np.log((second[1] + EPS) / (first[1] + EPS)),
                second[2] - first[2],
            ]
    if family in (None, "color"):
        red, green, blue = [normalized_residual(rgb[:, :, i]) for i in range(3)]
        cb_gradient = gradient(rgb[:, :, 2] - gray)
        cr_gradient = gradient(rgb[:, :, 0] - gray)
        values[12:] = [
            cosine(red, green), cosine(red, blue), cosine(green, blue),
            np.log((np.mean(cb_gradient + cr_gradient) / 2 + EPS) /
                   (np.mean(gradient(gray)) + EPS)),
            cosine(cb_gradient - cb_gradient.mean(), cr_gradient - cr_gradient.mean()),
        ]
    return values


def extract_features(image, long_edge=512, *, family=None):
    if family is not None and family not in FAMILIES:
        raise ValueError("Unknown statistics family")
    start = perf_counter()
    resized = bounded_rgb(image, long_edge)
    height, width = resized.shape[:2]
    side = min(256, height, width)
    positions = sorted({(y, x) for y in (0, height - side) for x in (0, width - side)})
    preprocessed = perf_counter()
    values = np.median([patch_statistics(resized[y:y + side, x:x + side], family)
                        for y, x in positions], axis=0)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite image statistics")
    return FeatureResult(values, len(positions), (preprocessed - start) * 1000,
                         (perf_counter() - preprocessed) * 1000)
