"""Source flip attribution with strict FAKE threshold and REAL equality."""

import pytest

from tools.diagnose_saved_thresholds import flip_details


def test_zero_ties_and_compensating_flips():
    rows = [{'src': str(i), 'label': label, 'score': score}
        for values in ((('REAL', 0.), ('FAKE', 1.)), (('REAL', 1.), ('FAKE', 0.)))
        for i, (label, score) in enumerate(values)]
    groups = {'before': [0, 1], 'after': [2, 3]}
    result = flip_details(rows, groups, [('before', 'after')])
    assert result[0]['exact_drop'] == '1'
    assert [v['transition'] for v in result[0]['flips']] == ['correct_to_wrong']*2
    assert [v['signed_degradation'] for v in result[0]['flips']] == [1., 1.]
    rows[3]['score'] = 2.
    assert flip_details(rows, groups, [('before', 'after')])[0]['exact_drop'] == '1/2'
    rows[3]['label'] = 'REAL'
    with pytest.raises(ValueError, match='Conflicting flip label'):
        flip_details(rows, groups, [('before', 'after')])
