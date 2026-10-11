"""Complete matched views with planted 10pp decline and malformed-pair refusal."""

from copy import deepcopy
import pytest
from palimpsest.evaluation.robust_views import evaluate_views


def planted_rows():
    rows=[]
    for variant in ('raw','q70'):
        for condition in ('original','physical'):
            for label in ('REAL','FAKE'):
                for i in range(10):
                    bad=variant=='q70' and condition=='physical' and label=='FAKE' and i<2
                    rows.append({'domain':'control','scene':'sample','src':label+str(i),'role':'selection','condition':condition,'variant':variant,'label':label,'score':0 if bad else (1 if label=='FAKE' else -1)})
    return rows


def test_known_view_loss_and_exact_boundary():
    result=evaluate_views(planted_rows(),variant_order=('raw','q70'),maximum_drop=.1)
    assert result['minimum_domain_ba']==.9 and result['maximum_domain_drop']==.1
    assert result['maximum_any_scene_drop']==.1 and result['development_all_scope_drop']
    assert not evaluate_views(planted_rows(),variant_order=('raw','q70'),maximum_drop=.099)['development_all_scope_drop']
    assert not result['independent_real_validation']


def test_malformed_source_views_refused():
    rows=planted_rows()
    with pytest.raises(ValueError):evaluate_views(rows+rows[:1],variant_order=('raw','q70'))
    with pytest.raises(ValueError):evaluate_views(rows[:-1],variant_order=('raw','q70'))
    bad=deepcopy(rows);bad[-1]['role']='fit'
    with pytest.raises(ValueError):evaluate_views(bad,variant_order=('raw','q70'))
    bad=deepcopy(rows);bad[-1]['label']='REAL'
    with pytest.raises(ValueError):evaluate_views(bad,variant_order=('raw','q70'))
