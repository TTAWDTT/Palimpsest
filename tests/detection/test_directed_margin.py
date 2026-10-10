"""Known directions, tiny-loss flips, graph edges and convex gradient controls."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.directed_margin import (
    DirectedPairs, directed_penalty, processing_pairs, fit_directed_source_risk, directed_objective)
from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk


def test_improvement_is_unpenalized_and_tiny_loss_can_flip_every_case():
    pair = DirectedPairs((0,), (1,), (1,), (1.,))
    value, grad = directed_penalty([1., 12.], [[-1.], [1.]], pair)
    assert value == 0 and np.array_equal(grad, [0, 0])
    value, grad = directed_penalty([1., 12.], [[1.], [-1.]], pair)
    assert value == 4 and np.array_equal(grad, [8, 0])
    for eps in (1e-3, 1e-6):
        values = np.array([[eps], [-eps]])
        penalty, _ = directed_penalty([1., 0.], values, pair)
        assert penalty == pytest.approx(4*eps*eps) and values[0, 0] > 0 >= values[1, 0]
    real = DirectedPairs((0,), (1,), (-1,), (1.,))
    assert directed_penalty([1., 0.], [[0.], [-1.]], real)[0] == 0  # Real tie stays correct.


def fixture():
    records = []; labels = []; features = []; groups = []
    for source, label in enumerate((0, 0, 1, 1)):
        for condition in ('original', 'a', 'b'):
            for variant in ('raw', 'q90', 'q60'):
                records.append(dict(domain='d', scene='s', src=str(source), role='fit',
                    condition=condition, variant=variant))
                labels.append(label); groups.append(source)
                features.append([(2*label-1)*(3 if condition == 'original' else 1)])
    return records, np.array(features, float), np.array(labels), np.array(groups)


def test_pair_graph_completeness_weight_and_training_role_refusal():
    records, x, y, groups = fixture()
    pairs = processing_pairs(records, y, np.ones(len(x)), groups, variants=('raw', 'q90', 'q60'))
    assert len(pairs.before) == 60 and pairs.cross_source_edges == 0
    assert sum(pairs.weights) == pytest.approx(1, abs=1e-14)
    for a, b, sign in zip(pairs.before, pairs.after, pairs.signed):
        assert y[a] == y[b] and sign == 2*y[a]-1
        assert records[a]['src'] == records[b]['src']
    unsigned = processing_pairs(records, y.astype(np.uint8), np.ones(len(x)), groups, variants=('raw', 'q90', 'q60'))
    assert unsigned.signed == pairs.signed
    for bad in (records[:-1], [{**r, 'role': 'selection'} for r in records]):
        n = len(bad)
        with pytest.raises(ValueError):
            processing_pairs(bad, y[:n], np.ones(n), groups[:n], variants=('raw', 'q90', 'q60'))


def test_directed_gradient_zero_reference_and_convex_fit():
    records, x, y, groups = fixture(); weights = np.ones(len(x))/len(x)
    pairs = processing_pairs(records, y, weights, groups, variants=('raw', 'q90', 'q60'))
    theta = np.array([.3, -.2])
    args = (x, 2*y-1, weights, groups, pairs)
    value, gradient, _ = directed_objective(theta, *args, ridge=.01, temperature=.1, strength=1)
    numerical = []
    for direction in np.eye(2):
        step = direction*1e-6
        numerical.append((directed_objective(theta+step, *args, ridge=.01, temperature=.1, strength=1)[0]
            -directed_objective(theta-step, *args, ridge=.01, temperature=.1, strength=1)[0])/2e-6)
    np.testing.assert_allclose(gradient, numerical, rtol=0, atol=2e-8)
    base, _ = fit_source_risk(x, y, weights, groups, feature_names=('known',))
    zero, _ = fit_directed_source_risk(x, y, weights, groups, pairs, strength=0, feature_names=('known',))
    np.testing.assert_array_equal(base.score(x), zero.score(x))
    unsigned, _ = fit_directed_source_risk(x, y.astype(np.uint8), weights, groups, pairs,
        strength=0, feature_names=('known',))
    np.testing.assert_array_equal(base.score(x), unsigned.score(x))
    rule, diagnostic = fit_directed_source_risk(x, y, weights, groups, pairs, strength=1, feature_names=('known',))
    assert diagnostic['maximum_absolute_gradient'] <= 1e-5 and diagnostic['objective'] <= diagnostic['initial_objective']
    assert np.array_equal(rule.score(x) > 0, y.astype(bool)) and value > 0
