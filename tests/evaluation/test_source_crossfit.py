"""Source leakage and decision-selection plants, before real CV."""

from fractions import Fraction

import pytest

from palimpsest.evaluation.source_crossfit import source_folds,crossfit_rates,choose_strength


def records():
    return [dict(domain='toy',scene='all',label=label,src=f'{label}/{i}',role='fit',
                 condition=condition,variant=variant,score=1. if label=='FAKE' else -1.)
            for label in ('REAL','FAKE') for i in range(4)
            for condition in ('original','processed') for variant in ('raw','jpeg')]


def test_source_fold_keeps_all_views_and_rejects_role():
    rows=records();a=source_folds(rows,folds=2)
    assert len(a)==8 and set(a.values())=={0,1}
    assert a==source_folds(list(reversed(rows)),folds=2)
    for label in ('REAL','FAKE'):
        assert sum(v==0 for (d,s),v in a.items() if s.startswith(label))==2
    with pytest.raises(ValueError):source_folds(rows+[rows[0]],folds=2)
    with pytest.raises(ValueError):source_folds([{**rows[0],'role':'selection'}],folds=2)
    with pytest.raises(ValueError):source_folds(rows,folds=5)


def test_oof_rate_and_wrong_selection():
    rows=records();r=crossfit_rates(rows,('raw','jpeg'))
    assert r['minimum_all_scope_ba']=='1' and r['maximum_all_scope_drop']=='0'
    for row in rows:
        if row['label']=='FAKE' and row['src']=='FAKE/0' and row['condition']=='processed':row['score']=-1.
    r=crossfit_rates(rows,('raw','jpeg'))
    assert Fraction(r['minimum_all_scope_ba'])==Fraction(7,8)
    assert Fraction(r['maximum_all_scope_drop'])==Fraction(1,8)
    candidates={'0':{'minimum_all_scope_ba':'9/10','maximum_all_scope_drop':'1/10'},
                '1':{'minimum_all_scope_ba':'17/20','maximum_all_scope_drop':'1/100'},
                '10':{'minimum_all_scope_ba':'3/4','maximum_all_scope_drop':'0'}}
    key,_=choose_strength(candidates);assert key=='1' and key!='10'
    with pytest.raises(ValueError):crossfit_rates(rows[:-1],('raw','jpeg'))
