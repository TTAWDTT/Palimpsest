"""Known probe and unit-response geometry; no target pixels or torch needed."""

import numpy as np
import pytest

from palimpsest.detection.representations.response_clip import gaussian_probe, response_feature
from palimpsest.detection.representations.intermediate_clip import shifted_midpoint


def test_known_response_geometry():
    a, b = np.zeros(1024), np.zeros(1024)
    a[0], b[1] = 1, 1
    u, v = shifted_midpoint(a), shifted_midpoint(b)
    result = response_feature(u, v)
    np.testing.assert_array_equal(result[:2], [.75, .25])
    assert result[-1] == .5
    np.testing.assert_array_equal(response_feature(u, u), np.r_[np.full(1024, .5), 0])
    np.testing.assert_array_equal(response_feature(u, shifted_midpoint(b*7)), result)


@pytest.mark.parametrize('value', [np.zeros(1024), np.ones(1023), np.full(1024, np.nan)])
def test_invalid_unit_response_refused(value):
    valid = np.full(1024, .5); valid[0] = 1
    with pytest.raises(ValueError):
        response_feature(valid, value)


def test_probe_channel_independence_and_mass():
    pixels = np.zeros((3, 224, 224), np.float32)
    pixels[1, 112, 112] = 1
    result = gaussian_probe(pixels)
    assert not result[[0, 2]].any()
    assert result[1, 112, 112] < 1
    assert abs(result.sum()-1) < 1e-6
    assert result[1, 112, 109] > 0 and result[1, 112, 108] == 0
    constant = np.full_like(pixels, 2.5)
    np.testing.assert_allclose(gaussian_probe(constant), constant, rtol=0, atol=1e-6)


@pytest.mark.parametrize('value', [np.zeros((3,224,224),np.float64),
    np.zeros((224,224,3),np.float32), np.full((3,224,224),np.nan,np.float32)])
def test_invalid_probe_refused(value):
    with pytest.raises(ValueError):
        gaussian_probe(value)
