"""Held-source isolation and known predictions for the reusable CV driver."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from palimpsest.evaluation.consistency_crossfit import crossfit_strengths


def fixture():
    records = [dict(domain='toy', scene='all', src=str(source), role='fit',
                    label='FAKE' if source >= 4 else 'REAL', condition=condition, variant=variant)
               for source in range(8) for condition in ('original', 'processed')
               for variant in ('raw', 'jpeg')]
    values = np.array([[int(r['src']), int(r['condition'] == 'processed')] for r in records], float)
    labels = (values[:, 0] >= 4).astype(int)
    return records, values, labels, np.ones(len(records)), values[:, 0].astype(int)


def test_crossfit_known_scores_and_training_source_isolation():
    records, x, y, w, s = fixture()
    calls = []

    def fit(train, labels, weights, groups, strength):
        trained = set(train[:, 0]); calls.append((trained, strength))
        assert len(trained) == 4 and set(groups) == set(range(4))
        assert np.array_equal(labels, (train[:, 0] >= 4).astype(int))

        def score(held):
            assert trained.isdisjoint(held[:, 0])
            out = np.where(held[:, 0] >= 4, 1., -1.)
            if strength == 0:
                out[(held[:, 0] == 4) & (held[:, 1] == 1)] = -1.
            return out
        return SimpleNamespace(score=score), {'trained_sources': len(trained)}

    results, chosen, folds = crossfit_strengths(records, x, y, w, s, (0, 1),
        ('raw', 'jpeg'), fit, fold_count=2)
    assert len(calls) == 4 and len(folds) == 8
    assert Fraction(results['0']['maximum_all_scope_drop']) == Fraction(1, 8)
    assert results['1']['minimum_all_scope_ba'] == '1' and chosen[0] == '1'
    assert all(len(r['scores']) == 32 for r in results.values())


def test_crossfit_rejects_numeric_group_merger_before_fit():
    records, x, y, w, s = fixture(); s[4:8] = s[0]
    with pytest.raises(ValueError, match='merges'):
        crossfit_strengths(records, x, y, w, s, (0, 1), ('raw', 'jpeg'),
            lambda *args: pytest.fail('Invalid grouping reached optimizer'), fold_count=2)
