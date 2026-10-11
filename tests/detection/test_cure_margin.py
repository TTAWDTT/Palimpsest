"""Author tie semantics must survive the shared strict margin evaluator."""

import math
import pytest

from palimpsest.detection.baselines.cure import CureFilePrediction, official_margin


def test_official_probability_threshold_and_tie():
    for probability in (0, .1, math.nextafter(.5, 0), .5, math.nextafter(.5, 1), .9, 1):
        assert (official_margin(probability) > 0) == (probability >= .5)
    prediction = CureFilePrediction(.5, 2, 3, 5)
    assert prediction.probability_fake == .5 and prediction.margin > 0


def test_invalid_probabilities_refused():
    for value in (-.01, 1.01, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='Invalid CuRe probability'):
            official_margin(value)
