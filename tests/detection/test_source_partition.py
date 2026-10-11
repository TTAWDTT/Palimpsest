"""Disjoint source-stage membership, stable ordering and numeric rule scope."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import palimpsest.detection.models.frozen_features.cure.split_rate_score as module
from palimpsest.detection.algorithms.source_partition import source_half_split
from palimpsest.detection.models.frozen_features.cure.quantile_score_consistency import QuantileScoreRule
from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from tests.detection.test_rate_penalty import fixture


def test_half_split_row_order_null_labels_and_odd_strata():
    records = [dict(domain='d', scene='s', src=str(i), role='fit', label='REAL') for i in range(6)]
    y = np.array([0, 0, 0, 1, 1, 1], np.uint8)
    mask = source_half_split(records, y)
    assert mask.sum() == 3 and 1 <= y[mask].sum() <= 2
    order = np.array([5, 1, 3, 0, 4, 2])
    np.testing.assert_array_equal(source_half_split([records[i] for i in order], y[order]), mask[order])
    records, y, _, _ = fixture(); mask = source_half_split(records, y)
    assert mask.sum() == 18
    assert all(len(set(mask[i:i+9])) == 1 for i in range(0, len(mask), 9))
    bad = deepcopy(records); bad[0]['role'] = 'threshold'
    with pytest.raises(ValueError, match='fit-only'):
        source_half_split(bad, y)
    bad = deepcopy(records); bad[1]['scene'] = 'other'
    with pytest.raises(ValueError, match='Conflicting'):
        source_half_split(bad, y)


def test_wrapper_passes_only_disjoint_stage_arrays(monkeypatch):
    records, y, w, sources = fixture(); x = np.c_[sources, y, np.zeros((len(y), 6))]
    names = ('raw0', 'raw1', 'clip0', 'clip1', 'r0', 'r1', 'r2', 'shape')
    host = module.SplitRateScoreFitter(names, variants=('raw', 'q90', 'q60'),
        legacy_dimensions=7, raw_dimensions=2, channels=2, filter_count=2)
    stages = {}
    def basis(values, labels, weights, groups, strength):
        stages['basis'] = set(values[:, 0]); assert strength == 0
        return SimpleNamespace(bank=None, center=(0.,)*5, scale=(1.,)*5,
            transform=lambda v: np.tile(v[:, :1], (1, 20))), {'scope': 'Artificial call-boundary witness'}
    def head(values, labels, weights, groups, panel, *, strength, **kwargs):
        assert kwargs['optimizer_ftol'] == 0
        stages['head'] = set(values[:, 0]); assert len(values) == 18 and len(set(groups)) == 2
        return StableRule(kwargs['feature_names'], (0.,)*20, (1.,)*20, (0.,)*20, 0., strength, .01), {
            'fit_records': len(values), 'sources': len(set(groups))}
    monkeypatch.setattr(host.provider, 'fit', basis)
    monkeypatch.setattr(module, 'fit_rate_source_risk', head)
    _, d = host.fit(x, y, w, sources, 10, records=records)
    assert stages['basis'].isdisjoint(stages['head'])
    assert stages['basis'] | stages['head'] == set(sources)
    assert d['fit_records'] == d['basis_fit_records'] == 18 and d['input_fit_records'] == 36
    changed = [{**r, 'src': '0_'+r['src']} for r in records]
    old_signature = d['stage_template_sha256']
    _, changed_d = host.fit(x, y, w, sources, 10, records=changed)
    # Same numeric context with changed SHA-ranked source identities must rebuild.
    assert old_signature != changed_d['stage_template_sha256']
    assert d['training_arrays_sha256'] == changed_d['training_arrays_sha256']
    assert stages['basis'] == {1., 3.} and stages['head'] == {0., 2.}


def test_actual_split_dtype_wrong_and_serialized_prediction(tmp_path):
    records, y, w, sources = fixture()
    x = np.array([[0., 0., 0., 0., 1/np.sqrt(2), 0., 1/np.sqrt(2),
        (2*int(label)-1)*(3 if r['condition'] == 'original' else 1)] for r, label in zip(records, y)])
    names = ('raw0', 'raw1', 'clip0', 'clip1', 'r0', 'r1', 'r2', 'shape')
    def fitter():
        return module.SplitRateScoreFitter(names, variants=('raw', 'q90', 'q60'),
            legacy_dimensions=7, raw_dimensions=2, channels=2, filter_count=2)
    host = fitter(); rule, d = host.fit(x, y, w, sources, 10, records=records)
    unsigned, _ = fitter().fit(x, y.astype(np.uint8), w, sources, 10, records=records)
    wrong, _ = host.fit(x, y, w, sources, 10, sources[::-1], records=records)
    np.testing.assert_array_equal(rule.score(x), unsigned.score(x))
    np.testing.assert_array_equal(rule.score(x), wrong.score(x))
    assert d['sources'] == d['basis_source_count'] == 2 and d['input_source_count'] == 4
    assert not set(d['basis_source_keys']) & set(d['readout_source_keys'])
    path = tmp_path/'rule.json'; rule.save(path)
    np.testing.assert_array_equal(QuantileScoreRule.load(path).score(x), rule.score(x))
    renamed = [{**r, 'src': '0_'+r['src']} for r in records]
    host.fit(x, y, w, sources, 10, records=renamed)
    # Here class-identical numeric basis arrays may share a child bank, while
    # the metadata-dependent stage templates must still be distinct.
    assert host.audit()['stage_templates'] == 2 and host.audit()['banks'] == 1


def test_training_record_boundary_is_pinned():
    from experiments.origin_detection.frozen_features.cure.split_rate_score.run_iteration import code_pins
    assert 'src/palimpsest/evaluation/training_fit.py' in {k.replace('\\', '/') for k in code_pins()}
