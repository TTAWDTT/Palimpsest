"""Exact brute counting verifies event sweeps and threshold equality."""

from fractions import Fraction

import pytest

from experiments.origin_detection.score_feasibility.audit_thresholds import sweep


def fixture(scores):
    # First half REAL, second half FAKE;two equally covered views.
    count = len(scores[0])
    rows = [{'domain': 'toy', 'src': str(i), 'label': 'REAL' if i < count // 2 else 'FAKE', 'score': score}
            for view in scores for i, score in enumerate(view)]
    groups = {'a': list(range(count)), 'b': list(range(count, 2 * count))}
    return rows, groups, [('a', 'b')]


def brute(rows, groups, pairs):
    states = [min(r['score'] for r in rows) - 1] + sorted({r['score'] for r in rows})
    count = 0
    minimum_drop = None
    maximum_ba = None
    for threshold in states:
        rates = {}
        for key, records in groups.items():
            class_rates = []
            for label in ('REAL', 'FAKE'):
                selected = [rows[i] for i in records if rows[i]['label'] == label]
                class_rates.append(Fraction(sum((r['score'] > threshold) == (label == 'FAKE') for r in selected), len(selected)))
            rates[key] = sum(class_rates) / 2
        low = min(rates.values())
        drop = max(rates[a] - rates[b] for a, b in pairs)
        if low >= Fraction(4, 5):
            minimum_drop = drop if minimum_drop is None else min(minimum_drop, drop)
        if drop <= Fraction(1, 50):
            maximum_ba = low if maximum_ba is None else max(maximum_ba, low)
        count += low >= Fraction(4, 5) and drop <= Fraction(1, 50)
    return count, minimum_drop, maximum_ba


def assert_expected(actual, expected):
    if actual != expected:
        raise ValueError('Known feasibility count differs')


@pytest.mark.parametrize('scores', [([-1, -1, 1, 1], [-1, 0, 0, 1]),
                                  ([-1, 0, 1, 1], [-1, 0, 1, 1]),
                                  ([0, 0, 0, 0], [0, 0, 0, 0])])
def test_tied_events_against_independent_fraction_brute(scores):
    args = fixture(scores)
    result = sweep(*args)
    count, drop, ba = brute(*args)
    assert_expected(result['feasible_event_states'], count)
    assert result['minimum_maximum_drop_under_ba80'] == (None if drop is None else str(drop))
    assert result['maximum_minimum_ba_under_drop2pp'] == (None if ba is None else str(ba))
    with pytest.raises(ValueError, match='Known feasibility'):
        assert_expected(result['feasible_event_states'], count + 1)


def test_exact_ba80_and_drop2_boundary():
    # Fifty records:80% correct, then78%, exactly1/50 decline but BA floor fails.
    a = [-1] * 20 + [1] * 5 + [1] * 20 + [-1] * 5
    b = a.copy()
    b[19] = 1
    result = sweep(*fixture((a, b)))
    assert result['zero_threshold'] == {'minimum_ba': '39/50', 'maximum_drop': '1/50'}
    exact = sweep(*fixture((a, a)))
    assert exact['zero_threshold']['minimum_ba'] == '4/5'
    assert exact['feasible_event_states'] == 1
    # 82% before,80% after:both floors pass at the exact2pp boundary.
    c = a.copy()
    c[20] = -1
    boundary = sweep(*fixture((c, a)))
    assert boundary['zero_threshold'] == {'minimum_ba': '4/5', 'maximum_drop': '1/50'}
    assert boundary['feasible_event_states'] == 1


def test_coverage_and_nonfinite_refusals():
    rows, groups, pairs = fixture(([-1, -1, 1, 1], [-1, -1, 1, 1]))
    rows[4]['src'] = 'different'
    with pytest.raises(ValueError, match='coverage'):
        sweep(rows, groups, pairs)
    rows[4]['src'] = '0'
    rows[0]['score'] = float('nan')
    with pytest.raises(ValueError, match='Invalid saved'):
        sweep(rows, groups, pairs)
