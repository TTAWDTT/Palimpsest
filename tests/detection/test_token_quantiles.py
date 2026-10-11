"""Same old statistics/different shape, transformations and blind-direction plant."""

import numpy as np
import pytest

from palimpsest.detection.representations.token_quantiles import projected_quantiles
from palimpsest.detection.representations.token_statistics import clipped_token_statistics
from palimpsest.detection.representations.token_covariance import projected_covariance


def test_old_statistics_miss_a_known_shape_difference():
    a = np.array([-2, -1, -1, 1, 1, 2], float)[:, None]+3
    b = np.array([-29/13, -1, -2/13, 2/13, 1, 29/13], float)[:, None]+3
    assert a.mean() == b.mean() == 3
    np.testing.assert_allclose(np.var(a), np.var(b), rtol=0, atol=1e-14)
    ca, ua, _ = clipped_token_statistics(a)
    cb, ub, _ = clipped_token_statistics(b)
    np.testing.assert_allclose(ca, cb, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(ua, ub)
    pa, _ = projected_covariance(a, np.ones((1, 1)))
    pb, _ = projected_covariance(b, np.ones((1, 1)))
    np.testing.assert_array_equal(pa, pb)
    qa, _ = projected_quantiles(a, np.ones((1, 1)))
    qb, _ = projected_quantiles(b, np.ones((1, 1)))
    assert np.max(np.abs(qa-qb)) > .2


def test_bag_transformations_constant_and_mad_fallback():
    x = np.array([[0, 2], [0, 2], [0, 2], [1, 5]], float)
    p = np.eye(2)
    q, d = projected_quantiles(x, p)
    assert d['quantile_rms_fallback_directions'] == 2
    np.testing.assert_allclose(q, projected_quantiles(x[[3, 1, 0, 2]], p)[0], atol=1e-13, rtol=0)
    np.testing.assert_allclose(q, projected_quantiles(8*x+[9, -7], p)[0], atol=1e-13, rtol=0)
    constant, diagnostics = projected_quantiles(np.ones((4, 2))*3, p)
    np.testing.assert_array_equal(constant, np.zeros(20))
    assert diagnostics['quantile_constant_directions'] == 2


def test_finite_projection_blindness_and_refusal():
    p = np.array([[1.], [0.]])
    a = np.array([[0., 0], [1, 0], [2, 0]])
    b = a.copy(); b[:, 1] = [-10, 2, 20]
    np.testing.assert_array_equal(projected_quantiles(a, p)[0], projected_quantiles(b, p)[0])
    assert not np.array_equal(a, b)  # Plant: same descriptor implies same bag is false.
    for invalid in (np.ones((1, 2)), np.full((3, 2), np.nan)):
        with pytest.raises(ValueError):
            projected_quantiles(invalid, p)
    with pytest.raises(ValueError):
        projected_quantiles(a, p*2)
