"""Old directions lack signal by construction; new raw direction separates it."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.semantic_score_margin import (
    four_score_terms, SemanticMarginFitter, SemanticScoreRule)


def test_four_dimensional_identity_and_invalid_terms():
    a = np.array([[1., 2., 3., 4.]]); b = np.array([[-1., 0., 2., 1.]])
    assert float(four_score_terms(a)[0]@four_score_terms(b)[0]) == pytest.approx(90.)
    bad = four_score_terms(b).copy(); bad[0, 6] = 0  # Deletes its nonzero (0,2) term.
    assert not np.isclose(float(four_score_terms(a)[0]@bad[0]), 90.)
    for invalid in (np.ones((2, 3)), np.full((1, 4), np.nan), np.full((1, 4), 1e200)):
        with pytest.raises(ValueError): four_score_terms(invalid)


def test_raw_direction_recovers_missing_signal_and_portable_path(tmp_path):
    # Abstract numeric features,not an assertion about a real image/token bag.
    common = np.array([0., 0., 1/np.sqrt(2), 0., 1/np.sqrt(2)])
    x = np.array([np.r_[raw, 0., common] for raw in (-1., -1., 1., 1.)])
    x = np.repeat(x, 3, axis=0); y = np.repeat([0, 0, 1, 1], 3)
    w = np.ones(len(x)); sources = np.repeat(np.arange(4), 3)
    names = ('raw0', 'raw1', 'clip0', 'clip1', 'r0', 'r1', 'r2')
    fitter = SemanticMarginFitter(names, raw_dimensions=2, channels=2, filter_count=2)
    zero, _ = fitter.fit(x, y, w, sources, 0); rule, diagnostic = fitter.fit(x, y, w, sources, .001)
    assert fitter.audit()['banks'] == 1 and rule.bank is zero.bank
    assert np.max(np.abs(rule.bank.old.transform(x[:, 2:]))) < 1e-10
    assert diagnostic['certificate']['passed'] and np.array_equal(rule.score(x) > 0, y.astype(bool))
    erased = x.copy(); erased[:, :2] = 0
    assert not np.array_equal(rule.score(erased) > 0, y.astype(bool))
    assert np.array_equal(rule.score(x), [rule.score([row])[0] for row in x])
    path = tmp_path/'rule.json'; rule.save(path)
    assert np.array_equal(SemanticScoreRule.load(path).score(x), rule.score(x))
    changed_raw = x.copy(); changed_raw[:, :2] *= 2
    another, _ = fitter.fit(changed_raw, y, w, sources, .001)
    assert fitter.audit()['banks'] == 2 and fitter.audit()['old_bank_builds'] == 1
    assert fitter.audit()['auxiliary_logistic_head_fits'] == 2
    assert np.array_equal(another.score(changed_raw) > 0, y.astype(bool))
    flipped, _ = fitter.fit(x, 1-y, w, sources, .001)
    assert fitter.audit()['banks'] == 3 and fitter.audit()['old_bank_builds'] == 2 and flipped.bank is not rule.bank
    assert fitter.audit()['auxiliary_logistic_head_fits'] == 3
    assert np.array_equal(flipped.score(x) > 0, (1-y).astype(bool))
    with pytest.raises(ValueError, match='Conflicting'):
        fitter.fit(x, y, w, sources, .001, np.zeros(len(sources), int))
