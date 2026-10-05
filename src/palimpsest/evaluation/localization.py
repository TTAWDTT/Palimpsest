"""Candidate localization quality against explicit region ground truth.

Matching maximizes valid match count first, total IoU second. Geometry metrics
do not establish whether a region contains a meaningful origin-classification
target. That requires an annotated target definition.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from palimpsest.contracts import Box, Region


def box_iou(a: Box, b: Box) -> float:
    area = max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(
        0, min(a.y1, b.y1) - max(a.y0, b.y0)
    )
    return area / (a.area + b.area - area)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.dtype != np.bool_ or b.dtype != np.bool_ or a.ndim != 2 or a.shape != b.shape:
        raise ValueError("IoU masks must be same-shape 2D bool arrays")
    union = np.count_nonzero(a | b)
    if union == 0:
        raise ValueError("IoU undefined for two empty masks")
    return float(np.count_nonzero(a & b) / union)


def evaluate_regions(
    predicted: tuple[Region, ...],
    targets: tuple[Region, ...],
    *,
    iou_threshold=0.5,
    use_masks=False,
) -> dict:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    if use_masks and any(r.mask is None for r in (*predicted, *targets)):
        raise ValueError("Mask evaluation requires masks on every region")
    matrix = np.zeros((len(predicted), len(targets)))
    for i, a in enumerate(predicted):
        for j, b in enumerate(targets):
            matrix[i, j] = (
                mask_iou(a.mask, b.mask) if use_masks else box_iou(a.box, b.box)
            )
    valid = matrix >= iou_threshold
    # Bonus larger than any possible total IoU change enforces count priority.
    utility = np.where(valid, matrix + min(matrix.shape, default=0) + 1, 0)
    rows, cols = linear_sum_assignment(utility, maximize=True)
    matches = [
        (int(i), int(j), float(matrix[i, j])) for i, j in zip(rows, cols) if valid[i, j]
    ]
    count = len(matches)
    return {
        "predictions": len(predicted),
        "targets": len(targets),
        "matched": count,
        "precision": count / len(predicted) if predicted else None,
        "recall": count / len(targets) if targets else None,
        "mean_matched_iou": float(np.mean([v for _, _, v in matches]))
        if matches
        else None,
        "iou_threshold": iou_threshold,
        "representation": "mask" if use_masks else "box",
        "matches": matches,
    }
