"""Known angles and a plant that computes cosine before undoing the shift."""

import numpy as np
import pytest

from palimpsest.detection.representations.query_sensitivity import shifted_cosine_drift


def test_known_angles_and_shift_plant():
    a = np.array([1., .5])  # Original unit (1,0).
    b = np.array([.5, 1.])  # Original unit (0,1).
    assert shifted_cosine_drift(a, a) == 0
    assert shifted_cosine_drift(a, b) == 1
    assert shifted_cosine_drift(a, 1-a) == 2
    wrong = 1-np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    assert abs(wrong-shifted_cosine_drift(a, b)) > .1


def test_rotation_symmetry_and_invalid_inputs():
    a, b = np.array([.8, .9]), np.array([.9, .2])
    assert shifted_cosine_drift(a, b) == shifted_cosine_drift(b, a)
    np.testing.assert_allclose(shifted_cosine_drift(a, b), shifted_cosine_drift(1-a, 1-b),
                               rtol=0, atol=2e-15)
    for invalid in (np.array([.5, .5]), np.array([np.nan, .5]), np.array([1.1, .5]), np.ones(3)):
        with pytest.raises(ValueError):
            shifted_cosine_drift(a, invalid)
