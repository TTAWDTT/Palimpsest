"""Known full-width mean storage and corrupt/schema refusal without Torch."""

import numpy as np
import pytest

from palimpsest.detection.representations.frozen_bfree import BFreeFeatures, FEATURE_NAMES, pack_mean
from experiments.origin_detection.bfree_readout.prepare_features import exact_feature


def test_known_crop_mean_storage_and_changed_coordinate():
    crops = np.zeros((5, 768), np.float32)
    crops[:, 0] = [1, 2, 3, 4, 5]
    crops[:, 1] = [-1, -2, -3, -4, -5]
    result = pack_mean(crops.mean(axis=0))
    expected = np.r_[3., -3., np.zeros(766)]
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64 and len(FEATURE_NAMES) == 768
    crops[:, 0] += 1
    assert not np.array_equal(pack_mean(crops.mean(axis=0)), result)


def test_invalid_mean_refused():
    for value in (np.zeros(767), np.zeros((1, 768)), np.full(768, np.inf), np.full(768, np.nan)):
        with pytest.raises(ValueError):
            pack_mean(value)


def test_actual_parity_checker_rejects_score_and_coordinate_corruption():
    clean = BFreeFeatures(np.zeros(768), 2., 0., 1.)
    exact_feature(clean, BFreeFeatures(np.zeros(768), 2., 0., 1.))
    with pytest.raises(ValueError):
        exact_feature(clean, BFreeFeatures(np.zeros(768), 2.001, 0., 1.))
    altered = np.zeros(768)
    altered[3] = .001
    with pytest.raises(ValueError):
        exact_feature(clean, BFreeFeatures(altered, 2., 0., 1.))
