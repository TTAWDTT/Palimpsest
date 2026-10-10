"""An independently specified nuisance inversion and its misleading-gate control."""

from dataclasses import replace

import numpy as np
import pytest

from palimpsest.detection.algorithms.conditional_residual import (
    ConditionalRule, basis_names, conditional_basis, gate_values,
)
from palimpsest.detection.algorithms.readouts.stable_rule import StableRule, fit_stable_rule
from experiments.origin_detection.conditional_residual.fit_rules import source_fold


def test_nuisance_inversion_and_wrong_gate(tmp_path):
    # Truth is sign(signal), but channel=1 reverses the observed signal. Metadata
    # is never supplied to score(): the second observable coordinate carries it.
    names = ('observed_signal', 'observable_nuisance')
    x = np.array([[-1, 0], [1, 0], [1, 1], [-1, 1]], float)
    labels = [0, 1, 0, 1]
    gate = StableRule(names, (0., 0.), (1., 1.), (0., 2.), -1., 0., .1)
    g = gate_values(gate, x)
    np.testing.assert_array_equal(g, [0, 0, 1, 1])
    basis = conditional_basis(x, g)
    readout, _ = fit_stable_rule(basis, labels, np.ones(4), np.zeros((1, 5)), [1.],
                               feature_names=basis_names(names), strength=0., ridge=.1)
    rule = ConditionalRule(gate, readout)
    assert list(rule.score(x) > 0) == [False, True, False, True]
    linear, _ = fit_stable_rule(x, labels, np.ones(4), np.zeros((1, 2)), [1.],
                               feature_names=names, strength=0., ridge=.1)
    np.testing.assert_array_equal(linear.score(x), np.zeros(4))
    wrong = replace(rule, gate=replace(gate, weights=(0., -2.), bias=1.))
    assert np.mean((wrong.score(x) > 0) == labels) == 0
    np.testing.assert_array_equal(rule.score(x), [rule.score([v])[0] for v in x])
    path = tmp_path/'rule.json'; rule.save(path)
    restored = ConditionalRule.load(path)
    assert restored.fingerprint == rule.fingerprint
    np.testing.assert_array_equal(restored.score(x), rule.score(x))


def test_gate_clipping_and_source_fold_controls():
    gate = StableRule(('x',), (0.,), (1.,), (1.,), 0., 0., .1)
    np.testing.assert_array_equal(gate_values(gate, [[-10], [0], [10]]), [0, .5, 1])
    for x, g in [([[0]], [1.1]), ([[np.nan]], [.5]), ([[0]], []), ([0], [.5])]:
        with pytest.raises(ValueError): conditional_basis(x, g)
    with pytest.raises(ValueError): ConditionalRule(gate, gate)
    # A source's variants/conditions are assigned by the same source string only.
    ids = [f'source-{n}' for n in range(30)]
    assert set(source_fold(s, 3) for s in ids) == {0, 1, 2}
    assert all(source_fold(s, 3) == source_fold(s, 3) for s in ids)
