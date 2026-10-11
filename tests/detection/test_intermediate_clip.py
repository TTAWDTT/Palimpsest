"""Known midpoint token storage; no encoder or target images required."""

import numpy as np
import pytest

from palimpsest.detection.representations.intermediate_clip import shifted_midpoint


def test_midpoint_unit_storage():
    values = np.zeros(1024)
    values[:2] = (3, -4)
    shifted = shifted_midpoint(values)
    np.testing.assert_allclose(shifted[:2], [.8, .1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(2*shifted-1, values/5, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(shifted, shifted_midpoint(values*2))
    for invalid in (np.zeros(1024), np.ones(1023), np.full(1024, np.nan)):
        with pytest.raises(ValueError):
            shifted_midpoint(invalid)
