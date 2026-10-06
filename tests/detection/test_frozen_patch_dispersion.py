import numpy as np
import pytest

from palimpsest.detection.representations.frozen_patch_dispersion import patch_dispersion


def test_hand_calculated_standard_deviation_and_gain():
    values = np.zeros((2, 384))
    values[0, 0], values[1, 1] = 1, 1
    expected = np.full(384, .5)
    expected[:2] = .5 + .5 / np.sqrt(2)
    assert np.array_equal(patch_dispersion(values), expected)
    assert np.array_equal(patch_dispersion(values * 7), expected)


def test_equal_mean_different_spread_and_fixed_token_permutation():
    a, b = np.zeros((2, 384)), np.zeros((2, 384))
    a[:, 0], b[:, 1] = [1, -1], [1, -1]
    assert np.array_equal(a.mean(axis=0), b.mean(axis=0))
    assert not np.array_equal(patch_dispersion(a), patch_dispersion(b))
    assert np.array_equal(patch_dispersion(a), patch_dispersion(a[::-1]))


def test_zero_spread_convention_and_invalid_inputs():
    values = np.ones((4, 384))
    assert np.array_equal(patch_dispersion(values), np.full(384, .5))
    for bad in (np.zeros((4, 384)), np.ones((1, 384)), np.ones((4, 383)),
                np.full((4, 384), np.nan)):
        with pytest.raises(ValueError):
            patch_dispersion(bad)
