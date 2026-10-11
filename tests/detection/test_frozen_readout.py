"""Before real fitting: independently specified XOR and broken-label controls."""

import csv
import json
import tomllib

import numpy as np
import pytest

from palimpsest.evaluation.features import validate_feature_cache, feature_views
from palimpsest.io.tables import read_rows
from experiments.origin_detection.frozen_readout.run_diagnostic import CONFIG, evaluate_readout, run
from experiments.origin_detection.frozen_readout.fit_readout import fit_readout
from experiments.origin_detection.ordinal_statistics.fit_rules import choose_threshold

NAMES = ('left', 'right')


def xor_table():
    rows = []
    for domain, scenes, conditions in [('rr',['all'],['original','transfer','redigital']),
                                       ('chimera',['cat','church','horse'],['original','mac_iphone','lg_blackfly'])]:
        for scene in scenes:
            for role in ('fit','threshold','selection'):
                for index in range(32):
                    a, b = (index//8)//2, (index//8)%2
                    label = 'FAKE' if a != b else 'REAL'
                    source = f'{domain}/{scene}/{role}/{index}'
                    for condition in conditions:
                        rows.append({'filename':source+'/'+condition,'src':source,'condition':condition,
                                     'domain':domain,'scene':scene,'role':role,'label':label,
                                     'sha256':'plant','variant':'raw','left':float(a),'right':float(b)})
    return rows


def config():
    with CONFIG.open('rb') as stream:
        return tomllib.load(stream)


def test_csv_full_path_xor_and_zero_null(tmp_path):
    rows = xor_table()
    path = tmp_path/'plant.csv'
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    read = validate_feature_cache(read_rows(path),rows,NAMES)
    result = run({'plant':(read,NAMES,{'generator':'independent XOR construction'})},config(),tmp_path/'out')
    labeled=result['results']['plant']['labeled']
    assert labeled['accuracy_gate_passed']
    # A single permutation can retain a four-pattern association by chance.
    # This deliberately exercises refusal; do not retune the seed to hide it.
    assert result['results']['plant']['source_shuffled_null']['accuracy_gate_passed']
    assert result['interpretation_refused']
    assert all(v['auc']==1 and v['balanced_accuracy_at_zero']==1 for v in labeled['selection_views'].values())
    stored=json.loads((tmp_path/'out/diagnostic.json').read_text(encoding='utf-8'))
    assert stored['results']['plant']['labeled']['threshold']==labeled['threshold']
    with pytest.raises(FileExistsError):
        run({'plant':(read,NAMES,{})},config(),tmp_path/'out')
    null=[{**r,'left':0.0,'right':0.0} for r in rows]
    result,rule=evaluate_readout(null,NAMES,config())
    assert not result['accuracy_gate_passed']
    assert all(v['auc']==0.5 for v in result['selection_views'].values())
    assert np.ptp(rule.score(np.zeros((20,2))))==0


@pytest.mark.parametrize('corruption',['duplicate','missing','label','role','nan'])
def test_feature_table_rejects_broken_identity(corruption):
    clean=xor_table();rows=[dict(r) for r in clean]
    validate_feature_cache(rows,clean,NAMES)
    if corruption=='duplicate':rows.append(dict(rows[0]))
    if corruption=='missing':rows.pop()
    if corruption=='label':rows[0]['label']='FAKE'
    if corruption=='role':rows[0]['role']='selection'
    if corruption=='nan':rows[0]['left']=float('nan')
    with pytest.raises(ValueError):validate_feature_cache(rows,clean,NAMES)


def test_no_selection_contact_and_readout_input_refusals():
    rows=xor_table();views=feature_views(rows,NAMES,'fit',processed=True)
    rule=fit_readout(views)
    probes=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
    assert (rule.score(probes)>0).tolist()==[False,True,True,False]
    # Arbitrary selection changes cannot enter the fitter or its threshold data.
    altered=[{**r,'left':1.0,'right':0.0,'label':'FAKE'} if r['role']=='selection' else r for r in rows]
    other=fit_readout(feature_views(altered,NAMES,'fit',processed=True))
    np.testing.assert_array_equal(rule.score(probes),other.score(probes))
    first=choose_threshold(rule,feature_views(rows,NAMES,'threshold',processed=True))
    second=choose_threshold(other,feature_views(altered,NAMES,'threshold',processed=True))
    assert first.threshold==second.threshold
    for bad in (np.zeros((1,3)),np.full((1,2),np.nan),np.zeros(2)):
        with pytest.raises(ValueError):rule.score(bad)
    broken=feature_views(rows,NAMES,'selection',processed=True)
    with pytest.raises(ValueError,match='fit-role'):fit_readout(broken)
    with pytest.raises(ValueError,match='eight'):fit_readout({})
