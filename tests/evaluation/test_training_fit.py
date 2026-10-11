"""Metadata alignment and exclusion, with unchanged legacy call signatures."""

import numpy as np
import pytest

from palimpsest.evaluation.training_fit import fit_training_rows
from palimpsest.evaluation.source_panel_crossfit import calibrated_panel_crossfit
from tests.evaluation.test_source_panel_crossfit import planted_panel
from types import SimpleNamespace


def test_legacy_and_record_aware_signatures():
    x = np.ones((2, 1)); records = [{'role': 'fit'}, {'role': 'fit'}]
    def legacy(a, b, c, d, e):
        return ('legacy', e)
    assert fit_training_rows(legacy, x, [0, 1], [1, 1], [0, 1], .1, records=records) == ('legacy', .1)
    def contextual(a, b, c, d, e, wrong=None, *, records):
        assert len(records) == len(a) and wrong == 'control'
        return records
    assert len(fit_training_rows(contextual, x, [0, 1], [1, 1], [0, 1], .1,
        records=records, record_aware=True, wrong='control')) == 2
    for bad in (records[:1], [{'role': 'threshold'}]*2):
        with pytest.raises(ValueError):
            fit_training_rows(legacy, x, [0, 1], [1, 1], [0, 1], .1, records=bad)


def test_source_held_fit_receives_only_aligned_training_context():
    panel, records, x, y, groups = planted_panel(); calls = []
    x = np.c_[x, np.arange(len(records))]
    row_indices = {tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')): i for i, r in enumerate(records)}
    def fit(values, labels, weights, sources, parameter, *, records):
        assert len(records) == len(values) == 6804
        ids = set(values[:, 0]); assert len(ids) == 756
        for row, value in zip(records, values):
            assert row['role'] == 'fit' and int(row['label'] == 'FAKE') == value[1]
            assert row_indices[tuple(row[k] for k in ('domain', 'src', 'condition', 'variant'))] == value[2]
        calls.append(ids)
        def score(query):
            assert ids.isdisjoint(query[:, 0])
            return 2*query[:, 1]-1
        return SimpleNamespace(score=score, threshold=0.), {'fit_records': len(values), 'sources': 756}
    def calibrate(rule, views):
        assert len(views) == 36
        assert all(calls[-1].isdisjoint(v[:, 0]) for _, v, _ in views.values())
        return rule, {'passed': True}
    results, chosen, _ = calibrated_panel_crossfit(records, x, y, np.ones(len(x)), groups,
        (.1,), ('id', 'class', 'row_index'), panel, fit, calibrate, record_aware=True)
    assert len(calls) == 5 and chosen[0] == '0.1'
    assert results['0.1']['minimum_all_scope_ba'] == '1' and results['0.1']['maximum_all_scope_drop'] == '0'
