"""Known rate groups, analytic derivative, and proxy/hard-rate discrepancy."""

from copy import deepcopy

import numpy as np
import pytest

from palimpsest.detection.algorithms.readouts.rate_penalty import training_rate_panel, rate_penalty, fit_rate_source_risk
from palimpsest.detection.models.frozen_features.cure.rate_score_readout import RateScoreFitter
from palimpsest.detection.models.frozen_features.cure.quantile_score_consistency import QuantileScoreFitter, QuantileScoreRule
from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk


def fixture():
    records, labels, groups = [], [], []
    for source, label in enumerate((0, 0, 1, 1)):
        for condition in ('original', 'a', 'b'):
            for variant in ('raw', 'q90', 'q60'):
                records.append(dict(domain='d', scene='s', src=str(source), role='fit',
                    condition=condition, variant=variant))
                labels.append(label); groups.append(source)
    return records, np.array(labels), np.ones(len(labels)), np.array(groups)


def test_panel_weights_and_refused_context():
    records, y, w, sources = fixture(); w[:18] = 3
    panel = training_rate_panel(records, y, w, sources, variants=('raw', 'q90', 'q60'))
    assert len(panel.keys) == 18 and len(panel.before) == 30
    np.testing.assert_allclose(panel.averaging@np.ones(len(y)), 1)
    np.testing.assert_allclose(panel.averaging@(y == 0), .5)
    bad = deepcopy(records); bad[0]['role'] = 'threshold'
    with pytest.raises(ValueError, match='training-only'):
        training_rate_panel(bad, y, w, sources, variants=('raw', 'q90', 'q60'))
    with pytest.raises(ValueError):
        training_rate_panel(records[:-1], y[:-1], w[:-1], sources[:-1], variants=('raw', 'q90', 'q60'))


def test_rate_gradient_and_proxy_not_hard_constraint():
    records, y, w, sources = fixture()
    panel = training_rate_panel(records, y, w, sources, variants=('raw', 'q90', 'q60'))
    z = np.random.default_rng(20261008).normal(size=(len(y), 2)); theta = np.array([.17, -.29, .11])
    value, gradient, _ = rate_penalty(theta, z, 2*y-1, panel)
    step = 1e-6; actual = []
    for i in range(len(theta)):
        delta = np.eye(len(theta))[i]*step
        actual.append((rate_penalty(theta+delta, z, 2*y-1, panel)[0]
                       - rate_penalty(theta-delta, z, 2*y-1, panel)[0])/(2*step))
    assert value > 0 and abs(gradient[-1]) > 1e-5
    np.testing.assert_allclose(gradient, actual, rtol=1e-5, atol=1e-8)
    z = (2*y-1)[:, None].astype(float)
    _, _, errors = rate_penalty([1e-5, 0], z, 2*y-1, panel)
    assert np.array_equal((z[:, 0]*1e-5 > 0), y.astype(bool))
    assert np.all(errors > .49)  # hard BA=1 yet soft BA near1/2.
    for invalid in ([np.nan, 0], [1e308, 1e308]):
        with pytest.raises(ValueError):
            rate_penalty(invalid, z, 2*y-1, panel)


def test_solver_zero_exact_and_nonzero_stationary():
    records, y, w, sources = fixture()
    panel = training_rate_panel(records, y, w, sources, variants=('raw', 'q90', 'q60'))
    x = (2*y-1)[:, None].astype(float)
    zero, _ = fit_rate_source_risk(x, y.astype(np.uint8), w, sources, panel, strength=0, feature_names=('s',))
    base, _ = fit_source_risk(x, y, w, sources, feature_names=('s',))
    np.testing.assert_array_equal(zero.score(x), base.score(x))
    rule, diagnostic = fit_rate_source_risk(x, y, w, sources, panel, strength=100,
        optimizer_ftol=0., feature_names=('s',))
    assert diagnostic['optimizer_ftol'] == 0
    assert diagnostic['maximum_absolute_gradient'] <= 1e-5
    assert diagnostic['minimum_fit_hard_ba_at_zero'] == 1
    assert np.array_equal(rule.score(x)>0, y.astype(bool))


def test_soft_zero_penalty_can_fail_hard_ba_and_directional_boundaries():
    records, y, sources, signed_margins = [], [], [], []
    for source in range(20):
        label = int(source >= 10)
        for condition in ('original', 'a', 'b'):
            for variant in ('raw', 'q90', 'q60'):
                records.append(dict(domain='d', scene='s', src=str(source), role='fit',
                    condition=condition, variant=variant))
                y.append(label); sources.append(source)
                signed_margins.append(-.001 if source % 10 < 3 else 3.)
    y, sources = np.array(y), np.array(sources)
    signed = 2*y-1; z = (signed*np.array(signed_margins))[:, None]
    panel = training_rate_panel(records, y, np.ones(len(y)), sources, variants=('raw', 'q90', 'q60'))
    penalty, _, rates = rate_penalty([1., 0.], z, signed, panel)
    hard_error = ((z[:, 0] > 0) != (y == 1)).astype(float)
    assert penalty == 0 and np.min(1-rates) > .8
    np.testing.assert_allclose(1-panel.averaging@hard_error, .7)
    # Zero score predicts REAL on every row, not a fractional hard decision.
    zero_scores = np.zeros(len(y)); correct = (zero_scores > 0) == (y == 1)
    assert correct[y == 0].all() and not correct[y == 1].any()
    np.testing.assert_allclose(rate_penalty([0., 0.], z, signed, panel)[2], .5)
    improve = z.copy(); degrade = z.copy()
    for i, row in enumerate(records):
        if row['condition'] == 'a' and int(row['src']) in (0, 10):
            improve[i, 0] = signed[i]*3.
        if row['condition'] == 'a' and int(row['src']) in (9, 19):
            degrade[i, 0] = -signed[i]*3.
    assert rate_penalty([1., 0.], improve, signed, panel)[0] == 0
    loss, _, errors = rate_penalty([1., 0.], degrade, signed, panel)
    assert loss > 0 and np.max(errors[list(panel.after)]-errors[list(panel.before)]) > .09


def test_wrapper_cold_dtype_wrong_identity_and_portable(tmp_path):
    records, y, w, sources = fixture()
    names = ('raw0', 'raw1', 'clip0', 'clip1', 'r0', 'r1', 'r2', 'shape')
    x = np.array([[0., 0., 0., 0., 1/np.sqrt(2), 0., 1/np.sqrt(2),
        (2*int(label)-1)*(3 if row['condition'] == 'original' else 1)] for row, label in zip(records, y)])
    options = dict(legacy_dimensions=7, raw_dimensions=2, channels=2, filter_count=2)
    def fitter():
        return RateScoreFitter(names, variants=('raw', 'q90', 'q60'), **options)
    with pytest.raises(ValueError, match='records'):
        fitter().fit(x, y, w, sources, 0)
    host = fitter(); zero, _ = host.fit(x, y, w, sources, 0, records=records)
    old, _ = QuantileScoreFitter(names, **options).fit(x, y, w, sources, 0)
    np.testing.assert_array_equal(zero.score(x), old.score(x))
    rule, _ = host.fit(x, y, w, sources, 10, records=records)
    unsigned, _ = fitter().fit(x, y.astype(np.uint8), w, sources, 10, records=records)
    wrong, d = host.fit(x, y, w, sources, 10, sources[::-1], records=records)
    np.testing.assert_array_equal(rule.score(x), unsigned.score(x))
    np.testing.assert_array_equal(rule.score(x), wrong.score(x))
    assert not d['wrong_source_affects_objective']
    path = tmp_path/'rule.json'; rule.save(path)
    np.testing.assert_array_equal(QuantileScoreRule.load(path).score(x), rule.score(x))
    assert host.audit()['rate_readout_fit_calls'] == 3
