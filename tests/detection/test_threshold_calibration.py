"""Independent binary threshold enumeration and unsafe-role controls."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.paired_stability import StableRule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold


def test_known_interval_and_strict_cut():
    rule = StableRule(('x',), (0.,), (1.,), (1.,), 0., 0., .1)
    rows = [{'role': 'threshold'}]*4
    rule, receipt = class_threshold(rule, {'plant': (rows, [[-2], [-1], [1], [2]], np.array([0, 0, 1, 1]))})
    assert rule.threshold == -1 and receipt['worst_calibration_class_accuracy'] == 1
    assert list(rule.score([[-1], [1]]) > rule.threshold) == [False, True]
    _, receipt = class_threshold(rule, {'constant': (rows, [[0]]*4, np.array([0, 0, 1, 1]))})
    assert receipt['worst_calibration_class_accuracy'] == 0
    for records, labels in [([{'role': 'selection'}]*4, np.array([0, 0, 1, 1])), (rows, np.zeros(4))]:
        with pytest.raises(ValueError): class_threshold(rule, {'bad': (records, [[0]]*4, labels)})
    with pytest.raises(ValueError): class_threshold(rule, {})


def test_two_views_against_independent_hand_enumeration():
    rule = StableRule(('x',), (0.,), (1.,), (1.,), 0., 0., .1)
    scores = [[-3, -.5, .5, 3], [-2, 1, 2, 4]]; labels = [0, 0, 1, 1]
    # Enumerate the union cuts by ordinary Python, not the calibration implementation.
    ranked = []
    for cut in [-4, -3, -2, -.5, .5, 1, 2, 3, 4]:
        rates = [[sum((s > cut) == bool(y) for s, y in zip(view, labels) if y == c)/2 for c in [0, 1]] for view in scores]
        ranked.append((min(c for v in rates for c in v), min(sum(v)/2 for v in rates), -abs(cut), -cut, cut))
    data = {str(i): ([{'role': 'threshold'}]*4, [[v] for v in values], np.array(labels)) for i, values in enumerate(scores)}
    calibrated, receipt = class_threshold(rule, data)
    assert calibrated.threshold == max(ranked)[-1]
    assert receipt['worst_calibration_class_accuracy'] == max(ranked)[0]
