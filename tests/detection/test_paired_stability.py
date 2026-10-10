"""Closed-form answers and a counterexample where stability erases truth."""

from dataclasses import replace
import json

import numpy as np
import pytest

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule, fit_stable_rule
from palimpsest.detection.algorithms.paired_stability import StableDetector
from experiments.origin_detection.paired_stability.fit_rules import paired_deltas


def test_scalar_closed_form_and_signal_erasure_counterexample(tmp_path):
    x = np.array([[-1.], [1.]])
    rules = []
    for strength in (0., 1., 10.):
        rule, diag = fit_stable_rule(x, [0, 1], [1, 1], [[2.]], [1], feature_names=('x',), strength=strength)
        assert rule.weights[0] == pytest.approx(1/(1.1+4*strength))
        assert diag['paired_score_drift'] == pytest.approx(4/(1.1+4*strength)**2)
        assert diag['objective'] <= 1 and diag['normal_equation_residual'] < 1e-12
        if strength: assert diag['paired_score_drift'] <= 1/strength
        rules.append(rule)
    # This pair changes the same direction as the label; low drift does not prove useful invariance.
    assert abs(rules[2].weights[0]) < abs(rules[0].weights[0])/30
    path = tmp_path/'rule.json'; rules[1].save(path)
    restored = StableRule.load(path)
    assert restored.fingerprint == rules[1].fingerprint
    np.testing.assert_array_equal(restored.score(x), rules[1].score(x))
    payload = json.loads(path.read_text()); payload['kind'] = 'forest'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): StableRule.load(path)
    with pytest.raises(ValueError): StableDetector(restored)
    for change in [{'scale': (0.,)}, {'strength': -1}, {'ridge': 0}, {'weights': (float('nan'),)},
                   {'threshold': float('nan')}]:
        with pytest.raises(ValueError): replace(rules[1], **change)


def test_orthogonal_known_signal_and_zero_label_signal():
    x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], float)
    rule, _ = fit_stable_rule(x, [0, 0, 1, 1], np.ones(4), [[0, 2], [0, -2]], [1, 1],
                              feature_names=('signal', 'nuisance'), strength=10.)
    np.testing.assert_allclose(rule.weights, [1/1.1, 0], atol=1e-14, rtol=0)
    assert list(rule.score(x)>0) == [False, False, True, True]
    null, _ = fit_stable_rule(x, [0, 1, 1, 0], np.ones(4), [[0, 2]], [1],
                              feature_names=('a', 'b'), strength=1.)
    np.testing.assert_array_equal(null.score(x), np.zeros(4))  # XOR has no linear contrast.
    np.testing.assert_array_equal(rule.score(x), [rule.score([v])[0] for v in x])
    with pytest.raises(ValueError): rule.score([[float('nan'), 0]])


def test_paired_group_identity_and_weight_control():
    rows = []
    for domain, conditions in [('rr', ['original', 'transfer', 'redigital']),
                               ('chimera', ['original', 'mac_iphone', 'lg_blackfly'])]:
        for role in ('fit', 'selection'):
            for c, value in zip(conditions, [0., 1., 3.]):
                for variant, shift in [('raw', 0.), ('jpeg', .5)]:
                    rows.append({'src': domain+'/'+role, 'domain': domain, 'scene': 'all', 'role': role,
                                 'label': 'REAL', 'condition': c, 'variant': variant, 'f': value+shift})
    delta, weights = paired_deltas(rows, 'fit', ('raw', 'jpeg'), ('f',))
    # Each source gives five deltas; exact finite multiset, irrespective of sorting.
    assert sorted(delta.ravel()) == [.5, .5, 1., 1., 1.5, 1.5, 3., 3., 3.5, 3.5]
    np.testing.assert_allclose(weights, np.full(10, .1), atol=0, rtol=0)
    for bad in [rows[:-1], rows+[rows[0]], [{**r, 'label': 'FAKE'} if i == 0 else r for i, r in enumerate(rows)]]:
        role = 'selection' if bad == rows[:-1] else 'fit'
        with pytest.raises(ValueError): paired_deltas(bad, role, ('raw', 'jpeg'), ('f',))
