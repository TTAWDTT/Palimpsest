"""Fixed averaging is portable, source complementary and not hard-rate safety."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from palimpsest.detection.algorithms.complementary_split import (
    ComplementarySplitFitter, ComplementarySplitRule, MEAN_NAMES, calibrate_complementary)
from palimpsest.detection.algorithms.paired_stability import StableRule
from palimpsest.detection.algorithms.quantile_score_consistency import QuantileScoreRule
from palimpsest.evaluation.calibrated_readout_campaign import CampaignMethod
from palimpsest.evaluation.training_fit import BudgetedRecordAwareCampaignMethod, bank_budgets
from tests.detection.test_rate_penalty import fixture


def test_score_average_can_reduce_ba_and_ties_are_real():
    vectors = (np.array([-1., 10., 1., 1.]), np.array([-1., -1., -10., 1.]))
    members = tuple(SimpleNamespace(bank=SimpleNamespace(feature_names=('x',)), threshold=0.,
        score=lambda _, v=v: v) for v in vectors)
    head = StableRule(MEAN_NAMES, (0., 0.), (1., 1.), (.5, .5), 0., 0., .01)
    rule = ComplementarySplitRule(members, head); labels = np.array([0, 0, 1, 1], bool)
    assert all(np.mean((v > 0) == labels) == .75 for v in vectors)
    assert np.mean((rule.score(np.zeros((4, 1))) > 0) == labels) == .5
    opposite = (members[0], SimpleNamespace(bank=members[0].bank, threshold=0., score=lambda _: -vectors[0]))
    scores = ComplementarySplitRule(opposite, head).score(np.zeros((4, 1)))
    np.testing.assert_array_equal(scores, 0)
    assert not (scores > 0).any()
    with pytest.raises(ValueError, match='Invalid fixed'):
        ComplementarySplitRule(members, replace(head, weights=(.25, .75)))


def test_two_component_data_budgets_and_old_payload_identity(tmp_path):
    records, y, w, sources = fixture()
    x = np.array([[0., 0., 0., 0., 1/np.sqrt(2), 0., 1/np.sqrt(2),
        (2*int(label)-1)*(3 if r['condition'] == 'original' else 1)] for r, label in zip(records, y)])
    names = ('raw0', 'raw1', 'clip0', 'clip1', 'r0', 'r1', 'r2', 'shape')
    host = ComplementarySplitFitter(names, variants=('raw', 'q90', 'q60'),
        legacy_dimensions=7, raw_dimensions=2, channels=2, filter_count=2)
    rule, d = host.fit(x, y.astype(np.uint8), w, sources, 10, records=records)
    assert d['fit_records'] == 36 and d['sources'] == 4
    assert d['mapped_dimensions'] == 2 and d['basis_count'] == 10
    assert [c['partition_complement'] for c in d['components']] == [False, True]
    assert d['components'][0]['basis_source_keys'] == d['components'][1]['readout_source_keys']
    np.testing.assert_array_equal(rule.score(x), rule.readout.score(rule.transform(x)))
    path = tmp_path/'mean.json'; rule.save(path)
    np.testing.assert_array_equal(rule.score(x), ComplementarySplitRule.load(path).score(x))
    member = rule.components[0]; expected = {'schema': 1, 'kind': 'quantile_score_consistency', 'rule': asdict(member)}
    assert member.to_payload() == expected
    np.testing.assert_array_equal(member.score(x), QuantileScoreRule.from_payload(expected).score(x))
    wrong, _ = host.fit(x, y, w, sources, 10, sources[::-1], records=records)
    np.testing.assert_array_equal(rule.score(x), wrong.score(x))
    views = {'toy': ([{**r, 'role': 'threshold'} for r in records], x, y)}
    fixed, cal = calibrate_complementary(rule, views, variants=('raw', 'q90', 'q60'))
    assert cal['joint_calibration_feasible'] and all(c.threshold == 0 for c in fixed.components)
    assert np.array_equal(fixed.score(x)>fixed.threshold, y.astype(bool))
    assert host.audit()['banks'] == 2 and host.audit()['ensemble_fit_calls'] == 2
    bad = json.loads(path.read_text()); bad['kind'] = 'other'; path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Unknown complementary'):
        ComplementarySplitRule.load(path)


def test_explicit_bank_budgets_preserve_old_defaults():
    fields = ((), (), (), lambda: {}, lambda: None, lambda: None, lambda: None)
    assert bank_budgets(CampaignMethod(*fields)) == (12, 1)
    method = BudgetedRecordAwareCampaignMethod(*fields, expected_banks=24, pilot_banks=2)
    assert bank_budgets(method) == (24, 2)
    for budgets in ((1, 2), (12, True), (24., 2), (0, 1)):
        with pytest.raises(ValueError, match='bank budget'):
            BudgetedRecordAwareCampaignMethod(*fields, expected_banks=budgets[0], pilot_banks=budgets[1])
