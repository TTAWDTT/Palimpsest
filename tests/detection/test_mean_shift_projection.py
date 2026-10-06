"""Planted retained signal, lost signal counterexample and panel refusal."""

from dataclasses import replace

import numpy as np
import pytest

from palimpsest.detection.algorithms.mean_shift_projection import MeanShiftProjection, fit_mean_shift
from palimpsest.detection.algorithms.source_view_risk import fit_source_risk


def plant(shift_axis=1):
    x = np.array([[-2., 0.], [-2., 0.], [2., 0.], [2., 0.]])
    x[[1, 3], shift_axis] += 3
    return x, np.array([0, 0, 1, 1]), np.ones(4), np.array([0, 0, 1, 1]), ['p']*4, ['raw', 'processed']*2


def fit(data):
    return fit_mean_shift(*data, feature_names=('class', 'nuisance'),
                          reference_view='raw', expected_views=('raw', 'processed'))


def test_known_projection_and_native_collapse(tmp_path):
    data = plant()
    mapper, diagnostic = fit(data)
    assert diagnostic['removed_rank'] == 1 and diagnostic['shift_vectors'] == 2
    transformed = mapper.transform(data[0])
    np.testing.assert_array_equal(transformed[:, 1], np.zeros(4))
    assert transformed[:, 0].tolist() == [-1., -1., 1., 1.]
    np.testing.assert_array_equal(transformed, np.vstack([mapper.transform([r]) for r in data[0]]))
    path = tmp_path/'map.json'
    mapper.save(path)
    np.testing.assert_array_equal(MeanShiftProjection.load(path).transform(data[0]), transformed)
    rule, _ = fit_source_risk(transformed, data[1], data[2], data[3], feature_names=mapper.output_names)
    assert (rule.score(transformed) > 0).tolist() == [False, False, True, True]
    collapsed = mapper.collapse(rule)
    np.testing.assert_allclose(collapsed.score(data[0]), rule.score(transformed), rtol=0, atol=1e-12)
    wrong = replace(mapper, directions=((1.,), (0.,)))
    assert not np.array_equal(wrong.transform(data[0]), transformed)


def test_zero_shift_and_class_loss_counterexample():
    data = list(plant())
    data[0][[1, 3], 1] = 0
    mapper, diagnostic = fit(data)
    assert diagnostic['removed_rank'] == 0
    np.testing.assert_array_equal(mapper.transform(data[0]), (data[0]-mapper.center)/mapper.scale)
    mapper, diagnostic = fit(plant(0))
    assert diagnostic['removed_rank'] == 1
    np.testing.assert_array_equal(mapper.transform(plant(0)[0]), np.zeros((4, 2)))


def test_corrupt_panels_and_nonfinite_refused():
    for index, value in [(1, [0, 1, 1, 1]), (4, ['p', 'q', 'p', 'p']), (5, ['raw']*4)]:
        data = list(plant())
        data[index] = value
        with pytest.raises(ValueError):
            fit(data)
    data = list(plant())
    data[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        fit(data)
    mapper, _ = fit(plant())
    with pytest.raises(ValueError):
        mapper.transform([[np.inf, 0]])
