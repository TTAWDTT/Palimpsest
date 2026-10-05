"""Known score/threshold and identity mismatch for the pixel-readout adapter."""

from types import SimpleNamespace
import numpy as np
import pytest
from palimpsest.contracts import Origin
from palimpsest.detection.algorithms.paired_stability import StableRule
from palimpsest.detection.representations.readout import FeatureReadoutDetector


def test_known_readout_and_threshold_boundary():
    rule = StableRule(('known',), (1.,), (2.,), (3.,), .5, 0., .1, threshold=2.)
    value = SimpleNamespace(values=np.array([2.]), preprocess_ms=0., statistics_ms=0.)
    detector = FeatureReadoutDetector(lambda image: value, rule, feature_names=('known',), name='control')
    prediction = detector.predict(np.zeros((2, 3, 3), np.uint8))
    assert prediction.score == 2. and prediction.margin == 0. and prediction.origin == Origin.NATURAL
    value.values[0] = 4.
    assert detector.predict(np.zeros((2, 3, 3), np.uint8)).score == 5.
    with pytest.raises(ValueError): FeatureReadoutDetector(lambda image: value, rule, feature_names=('wrong',), name='control')
    with pytest.raises(ValueError): detector.predict(np.zeros((2, 3, 3), float))
