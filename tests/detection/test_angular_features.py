"""Known radial/orthogonal plants, corruption, and conventional fit composition."""

import math

import numpy as np
import pytest

from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
from palimpsest.detection.representations.angular_features import AngularFeatures, paired_drift


def test_direction_radius_and_saved_single_query(tmp_path):
    x = np.array([[3., 4.], [3/8, 4/8], [24., 32.], [-3., -4.]])
    mapper = AngularFeatures(('a', 'b'))
    values = mapper.transform(x)
    np.testing.assert_array_equal(values, [[.6, .8], [.6, .8], [.6, .8], [-.6, -.8]])
    np.testing.assert_array_equal(values, np.vstack([mapper.transform([r]) for r in x]))
    path = tmp_path/'map.json'
    mapper.save(path)
    np.testing.assert_array_equal(AngularFeatures.load(path).transform(x), values)
    radius = AngularFeatures(('a', 'b'), True).transform(x)
    np.testing.assert_allclose(radius[:, -1], [math.log(5), math.log(5/8), math.log(40), math.log(5)], rtol=0, atol=1e-15)
    assert not np.array_equal(mapper.transform([[4., 3.]]), values[:1])
    assert not np.array_equal(radius[:1], radius[2:3])


def test_known_drift_and_invalid_inputs():
    result = paired_drift([[3., 4.], [1., 0.]], [[6., 8.], [0., 2.]])
    for key, expected in [('norm_ratio', [2, 2]), ('cosine', [1, 0]),
                          ('best_scalar', [2, 0]), ('nonradial_relative', [0, 1])]:
        np.testing.assert_array_equal(result[key], expected)
    mapper = AngularFeatures(('a', 'b'))
    for invalid in ([[0., 0.]], [[np.nan, 1]], [[1.]], []):
        with pytest.raises(ValueError):
            mapper.transform(invalid)
    with pytest.raises(ValueError):
        paired_drift([[1., 0.]], [[1., 0.], [2., 0.]])
    with pytest.raises(ValueError):
        AngularFeatures(('a', 'a'))


def test_planted_direction_classifier_and_scale_invariance():
    x = np.array([[-3., 4.], [-6., 8.], [3., 4.], [6., 8.]])
    mapper = AngularFeatures(('a', 'b'))
    z = mapper.transform(x)
    for temperature in (0, .1):
        rule, diagnostic = fit_source_risk(z, [0, 0, 1, 1], np.ones(4), np.array([0, 0, 1, 1]),
                                          feature_names=mapper.output_names, temperature=temperature)
        assert (rule.score(z) > 0).tolist() == [False, False, True, True]
        np.testing.assert_array_equal(rule.score(mapper.transform(x*8)), rule.score(z))
        assert diagnostic['fit_records'] == 4 and diagnostic['sources'] == 2
