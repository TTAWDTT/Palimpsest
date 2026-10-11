"""Known vector composition and invalid coverage; no pretrained inference."""

import numpy as np
import pytest

from palimpsest.detection.representations.feature_pair import FrozenFeaturePair
from palimpsest.detection.representations.frozen_clip import ClipFeatures


class KnownEncoder:
    provenance = {'scope':'Synthetic control only'}

    def __init__(self,values):
        self.values = np.asarray(values,float)

    def extract(self,image):
        return ClipFeatures(self.values.copy(),0,0)


def test_known_pair_and_component_dimension():
    pair = FrozenFeaturePair(KnownEncoder([.2,.3]),KnownEncoder([.7]),
        first_names=('a','b'),second_names=('c',))
    result = pair.extract(np.zeros((2,2,3),np.uint8))
    np.testing.assert_array_equal(result.values,[.2,.3,.7])
    assert result.statistics_ms >= 0
    pair.second = KnownEncoder([.7,.8])
    with pytest.raises(ValueError,match='dimension'):
        pair.extract(None)


def test_feature_pair_duplicate_names_refused():
    with pytest.raises(ValueError,match='distinct'):
        FrozenFeaturePair(KnownEncoder([1]),KnownEncoder([2]),first_names=('a',),second_names=('a',))
