"""Known missing raw signal and smooth-head pipeline/control identities."""

from dataclasses import replace

import numpy as np

from palimpsest.detection.models.frozen_features.cure.semantic_score_consistency import SemanticConsistencyFitter
from palimpsest.detection.models.frozen_features.cure.semantic_score_margin import SemanticScoreRule, calibrate_semantic_score


def test_smooth_semantic_head_and_calibration_route(tmp_path):
    common = [0., 0., 1/np.sqrt(2), 0., 1/np.sqrt(2)]
    x = np.array([np.r_[raw, 0., common] for raw in (-1., -1., 1., 1.)])
    x = np.repeat(x, 3, axis=0); y = np.repeat([0, 0, 1, 1], 3); w = np.ones(len(x)); s = np.repeat(np.arange(4), 3)
    fitter = SemanticConsistencyFitter(('raw0', 'raw1', 'clip0', 'clip1', 'r0', 'r1', 'r2'),
        raw_dimensions=2, channels=2, filter_count=2)
    zero, _ = fitter.fit(x, y, w, s, 0); rule, d = fitter.fit(x, y, w, s, 1)
    assert zero.bank is rule.bank and fitter.audit()['initialization_lp_fits'] == 1
    assert fitter.audit()['source_consistency_fit_calls'] == 2
    assert d['maximum_absolute_gradient'] <= 1e-5 and np.array_equal(rule.score(x) > 0, y.astype(bool))
    assert np.array_equal(rule.score(x), [rule.score([row])[0] for row in x])
    erased = x.copy(); erased[:, :2] = 0
    assert not np.array_equal(rule.score(erased) > 0, y.astype(bool))
    path = tmp_path/'rule.json'; rule.save(path)
    assert np.array_equal(SemanticScoreRule.load(path).score(x), rule.score(x))
    def calibrate(head, views):
        rows, mapped, labels = views['plant']
        assert np.array_equal(mapped, rule.transform(x)) and np.array_equal(labels, y)
        assert all(r['role'] == 'threshold' for r in rows)
        return replace(head, threshold=.123), {'passed': True}
    fixed, diagnostic = calibrate_semantic_score(rule, {'plant': ([{'role': 'threshold'}]*len(x), x, y)}, calibrate)
    assert diagnostic['passed'] and fixed.threshold == .123 and np.array_equal(fixed.score(x), rule.score(x))
    flipped, _ = fitter.fit(x, 1-y, w, s, 1)
    assert np.array_equal(flipped.score(x) > 0, (1-y).astype(bool)) and fitter.audit()['banks'] == 2
