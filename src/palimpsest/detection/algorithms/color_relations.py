"""Compact joint RGB statistics; a declared RTPO-inspired adaptation.

Strict product dominance means all components <= and at least one <. Ties
are counted separately. Fixed-grid monotone invariance of rank comparisons
does not survive arbitrary color mixing, resampling, clipping, or noise.
"""

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import cv2
import numpy as np

from .residual_statistics.features import resize256, quantize

SCALES = (1, 2, 4)
OFFSETS = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x)
TRIPLES = tuple(product(range(-2, 3), repeat=3))
ORBITS = tuple(sorted({min(t, tuple(-v for v in t)) for t in TRIPLES}))
LOOKUP = np.array([ORBITS.index(min(t, tuple(-v for v in t))) for t in TRIPLES])
ORDER_NAMES = tuple(f'color_order/{s}/{i}' for s in SCALES for i in range(45))
TIE_NAMES = tuple(f'color_ties/{s}/{i}' for s in SCALES for i in range(9))
JOINT_NAMES = tuple(f'color_residual/{s}/{i}' for s in SCALES for i in range(63))
FEATURE_NAMES = ORDER_NAMES + TIE_NAMES + JOINT_NAMES


def product_counts(rgb):
    """Dominated/incomparable/tied neighbor counts for every valid 3x3 center."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 3 or not np.isfinite(rgb).all():
        raise ValueError('Expected finite RGB grid with sides >=3')
    center = rgb[1:-1, 1:-1]
    dominated = np.zeros(center.shape[:2], np.int16)
    incomparable = np.zeros_like(dominated); tied = np.zeros_like(dominated)
    h, w = center.shape[:2]
    for dy, dx in OFFSETS:
        neighbor = rgb[1+dy:1+dy+h, 1+dx:1+dx+w]
        lower = np.all(neighbor <= center, axis=2)
        upper = np.all(center <= neighbor, axis=2)
        equal = np.all(neighbor == center, axis=2)
        dominated += lower & ~equal
        incomparable += ~lower & ~upper
        tied += equal
    return dominated, incomparable, tied


def order_histograms(rgb):
    dominated, incomparable, tied = product_counts(rgb)
    codes = incomparable + dominated*(19-dominated)//2
    order = np.bincount(codes.ravel(), minlength=45).astype(float)
    ties = np.bincount(tied.ravel(), minlength=9).astype(float)
    return order/order.sum(), ties/ties.sum()


def cross_counts(quantized):
    """Same-location RGB triple counts, folding only simultaneous sign reversal."""
    q = np.asarray(quantized)
    if (q.ndim != 3 or q.shape[2] != 3 or not q.size or q.dtype.kind not in 'iu'
            or np.any(q < -2) or np.any(q > 2)):
        raise ValueError('Expected integer RGB residuals in [-2,2]')
    q = q.astype(np.int16)+2
    codes = 25*q[..., 0]+5*q[..., 1]+q[..., 2]
    return np.bincount(LOOKUP[codes].ravel(), minlength=63)


def joint_histogram(rgb):
    rgb = np.asarray(rgb, np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 3 or not np.isfinite(rgb).all():
        raise ValueError('Expected finite RGB residual input')
    middle = rgb[1:-1, 1:-1]
    horizontal = rgb[1:-1, :-2]+rgb[1:-1, 2:]-2*middle
    vertical = rgb[:-2, 1:-1]+rgb[2:, 1:-1]-2*middle
    counts = sum((cross_counts(np.stack([quantize(plane[..., c]) for c in range(3)], axis=2))
                  for plane in (horizontal, vertical)))
    return counts/counts.sum()


@dataclass(frozen=True)
class ColorFeatures:
    values: np.ndarray
    preprocess_ms: float
    statistics_ms: float


def extract_colors(image):
    start = perf_counter()
    rgb = resize256(image)
    if min(rgb.shape[:2]) < 20:
        rgb = cv2.copyMakeBorder(rgb, 0, max(0, 20-rgb.shape[0]), 0, max(0, 20-rgb.shape[1]),
                                 cv2.BORDER_REFLECT_101)
    ready = perf_counter()
    pyramids = [rgb.astype(np.float32)]
    for scale in SCALES[1:]:
        pyramids.append(cv2.resize(rgb.astype(np.float32), (max(5, rgb.shape[1]//scale), max(5, rgb.shape[0]//scale)),
                                   interpolation=cv2.INTER_AREA))
    orders, ties, joint = [], [], []
    for p in pyramids:
        order, tie = order_histograms(p)
        orders.append(order); ties.append(tie); joint.append(joint_histogram(p))
    values = np.concatenate(orders+ties+joint)
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError('Color feature schema/numerics differ')
    return ColorFeatures(values, 1000*(ready-start), 1000*(perf_counter()-ready))
