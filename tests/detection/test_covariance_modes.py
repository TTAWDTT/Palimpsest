"""Planted covariance signal with exact deployment and invalid-state controls."""

from dataclasses import replace

import numpy as np
import pytest

from palimpsest.detection.algorithms.covariance_modes import CovarianceModes, fit_covariance_modes


def test_planted_variance_and_deployment_identity(tmp_path):
    x = np.array([[-.5, 0], [.5, 0], [-2, 0], [2, 0]], float)
    rows = np.repeat(x, 2, axis=0)
    y, sources = np.repeat([0, 0, 1, 1], 2), np.repeat(np.arange(4), 2)
    mapper, diagnostic = fit_covariance_modes(rows, y, np.ones(8), sources,
        feature_names=('signal', 'constant'), count=1)
    q = mapper.transform(x, quadratic_only=True)[:, 0]
    assert q[2] == q[0]*16 and q[3] == q[1]*16 and q[0] > 0
    assert diagnostic['whitener_residual_inf'] < 1e-12
    assert diagnostic['fit_sources'] == 4 and diagnostic['fit_records'] == 8
    assert mapper.directions[1][0] == 0
    full = mapper.transform(x)
    np.testing.assert_array_equal(full[:, :2], x)
    np.testing.assert_array_equal(full, np.vstack([mapper.transform([row]) for row in x]))
    destination = tmp_path / 'modes.json'
    mapper.save(destination)
    np.testing.assert_array_equal(CovarianceModes.load(destination).transform(x), full)
    wrong = replace(mapper, directions=((0.,), (1.,)))
    assert not np.array_equal(wrong.transform(x), full)


def test_conflicting_source_and_invalid_inputs_refused():
    x, y = np.array([[-1., 0], [1., 0], [-2., 0], [2., 0]]), [0, 0, 1, 1]
    with pytest.raises(ValueError, match='Conflicting'):
        fit_covariance_modes(x, y, np.ones(4), [0, 1, 0, 1], feature_names=('a', 'b'), count=1)
    for count in (0, 3):
        with pytest.raises(ValueError):
            fit_covariance_modes(x, y, np.ones(4), np.arange(4), feature_names=('a', 'b'), count=count)
    mapper, _ = fit_covariance_modes(x, y, np.ones(4), np.arange(4), feature_names=('a', 'b'), count=1)
    with pytest.raises(ValueError):
        mapper.transform([[np.nan, 0]])
