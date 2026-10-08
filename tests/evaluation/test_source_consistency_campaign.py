"""Preserve valid OOF controls but reject changed folds and paired scores."""

from copy import deepcopy
from fractions import Fraction

import pytest

from palimpsest.evaluation.source_crossfit import source_folds, crossfit_rates
from palimpsest.evaluation.source_consistency_campaign import validate_prior_oof


def test_reused_oof_known_decline_and_corruption_refusals():
    variants = ('raw','jpeg')
    rows = [dict(domain='toy',scene='all',src=f'{label}/{i}',role='fit',label=label,
                 condition=condition,variant=variant,score=1. if label=='FAKE' else -1.)
            for label in ('REAL','FAKE') for i in range(4)
            for condition in ('original','processed') for variant in variants]
    folds = source_folds(rows,folds=2);labels = [int(r['label']=='FAKE') for r in rows]
    for r in rows:
        r['fold'] = folds[r['domain'],r['src']]
        if r['src']=='FAKE/0' and r['condition']=='processed':r['score']=-1.
    result = {**crossfit_rates(rows,variants),'scores':rows}
    actual = validate_prior_oof(result,rows,labels,folds,variants)
    assert Fraction(actual['maximum_all_scope_drop'])==Fraction(1,8)
    bad = deepcopy(result);bad['scores'][0]['score']*= -1
    with pytest.raises(ValueError,match='arithmetic'):
        validate_prior_oof(bad,rows,labels,folds,variants)
    bad = deepcopy(result);bad['scores'][0]['fold'] = 1-bad['scores'][0]['fold']
    with pytest.raises(ValueError,match='identity'):
        validate_prior_oof(bad,rows,labels,folds,variants)
