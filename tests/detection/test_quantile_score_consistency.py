"""Known missing signal, old covariance layout and portable five-score readout."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.quantile_score_consistency import (
    five_score_terms, QuantileScoreFitter, QuantileScoreRule)


def test_five_score_expansion_and_zero_shape_terms():
    a = np.array([[1., 2, 3, 4, 5]]); b = np.array([[2., -1, 1, 0, -2]])
    dot = float(a[0]@b[0])
    assert float(five_score_terms(a)[0]@five_score_terms(b)[0]) == pytest.approx(dot+dot*dot, abs=1e-12)
    zero = a.copy(); zero[0, 4] = 0
    terms = five_score_terms(zero)[0]
    assert terms[4] == 0 and all(terms[5+k] == 0 for k, (i, j) in enumerate(
        (i, j) for i in range(5) for j in range(i, 5)) if j == 4)
    for bad in (np.zeros((2, 4)), np.full((1, 5), np.nan), np.full((1, 5), 1e200)):
        with pytest.raises(ValueError):
            five_score_terms(bad)


def test_shape_recovers_planted_signal_and_reuses_legacy(tmp_path):
    old = [0., 0., 0., 0., 1/np.sqrt(2), 0., 1/np.sqrt(2)]
    x = np.array([np.r_[old, signal] for signal in (-1., -1., 1., 1.)])
    x = np.repeat(x, 3, axis=0); y = np.repeat([0, 0, 1, 1], 3)
    weights = np.ones(len(x)); sources = np.repeat(np.arange(4), 3)
    names = ('raw0', 'raw1', 'clip0', 'clip1', 'r0', 'r1', 'r2', 'shape')
    fitter = QuantileScoreFitter(names, legacy_dimensions=7, raw_dimensions=2, channels=2, filter_count=2)
    zero, _ = fitter.fit(x, y, weights, sources, 0)
    rule, diagnostic = fitter.fit(x, y, weights, sources, 1)
    assert rule.bank is zero.bank and diagnostic['maximum_absolute_gradient'] <= 1e-5
    assert np.max(np.abs(rule.bank.transform(x)[:, :4])) < 1e-10
    assert np.array_equal(rule.score(x) > 0, y.astype(bool))
    assert np.array_equal(rule.score(x), [rule.score([row])[0] for row in x])
    path = tmp_path/'rule.json'; rule.save(path)
    assert np.array_equal(QuantileScoreRule.load(path).score(x), rule.score(x))
    changed = x.copy(); changed[:, -1] = 0
    masked, _ = fitter.fit(changed, y, weights, sources, 1)
    assert not np.array_equal(masked.score(changed) > 0, y.astype(bool))
    audit = fitter.audit()
    assert audit['banks'] == 2 and audit['shape_head_fits'] == 2 and audit['legacy_full_array_banks'] == 1
    assert audit['initialization_lp_fits'] == 1 and audit['source_consistency_fit_calls'] == 3
