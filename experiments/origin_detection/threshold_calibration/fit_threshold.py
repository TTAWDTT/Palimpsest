"""A single global threshold, maximizing the worst per-view class accuracy."""

from dataclasses import replace

import numpy as np


def class_threshold(rule, view_data):
    if not view_data: raise ValueError('Empty threshold views')
    scored = []
    for records, values, labels in view_data.values():
        if any(r['role'] != 'threshold' for r in records) or set(labels.tolist()) != {0, 1}:
            raise ValueError('Threshold role/classes differ')
        scores = rule.score(values)
        if not np.isfinite(scores).all(): raise ValueError('Nonfinite calibration scores')
        scored.append((scores, labels))
    unique = np.unique(np.concatenate([s for s, _ in scored]))
    best = None
    for t in np.r_[np.nextafter(unique[0], -np.inf), unique]:
        correct = [[float(((s > t) == y.astype(bool))[y == label].mean()) for label in (0, 1)] for s, y in scored]
        ranked = (min(min(c) for c in correct), min(sum(c)/2 for c in correct), -abs(float(t)), -float(t), float(t))
        if best is None or ranked > best: best = ranked
    return replace(rule, threshold=best[-1]), {'worst_calibration_class_accuracy': best[0],
                                              'worst_calibration_ba': best[1], 'views': len(scored)}
