"""Supervised covariance directions learned only within each training fold."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.csp_readout import fit_csp_source,calibrate_csp_rule,CSPRule
from palimpsest.evaluation.source_consistency_campaign import run_source_consistency
from palimpsest.evaluation.source_crossfit import source_folds
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR,REPO_ROOT
from experiments.origin_detection.cure_token_covariance.prepare_features import (
    OUTPUT as PARENT,code_pins as input_pins,parent_data,audit_cache)
from experiments.origin_detection.cure_token_covariance.run_iteration import COMBINED_NAMES as NAMES
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS,audit_receipt
from experiments.origin_detection.token_consistency_crossfit.run_iteration import (
    CONFIG,STRENGTHS,class_threshold,score_rule,wrong_sources,null_intervals,write_json)

OUTPUT=WORK_DIR/'robust_statistics/token_csp_consistency'


def code_pins():
    pins=input_pins();paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    paths+=[REPO_ROOT/p for p in (
        'src/palimpsest/detection/algorithms/csp_readout.py',
        'src/palimpsest/detection/algorithms/readouts/consistent_source_risk.py',
        'src/palimpsest/evaluation/source_consistency_campaign.py',
        'src/palimpsest/evaluation/consistency_crossfit.py','src/palimpsest/evaluation/source_crossfit.py',
        'tests/detection/test_csp_readout.py','tests/detection/test_consistent_source_risk.py',
        'tests/evaluation/test_source_crossfit.py','tests/evaluation/test_consistency_crossfit.py',
        'tests/evaluation/test_source_consistency_campaign.py','tools/audit_source_crossfit.py')]
    pins.update({str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)})
    return pins


def inputs():
    receipt=audit_receipt(PARENT/'features.json')
    if (receipt['records']!=15120 or not receipt['all_parent_prefix_exact']
            or file_sha256(PARENT/'vectors.npy')!=receipt['vectors_sha256']
            or file_sha256(PARENT/'metadata.csv')!=receipt['metadata_sha256']):
        raise ValueError('Signed covariance features changed')
    gate=json.loads((PARENT/'pre_fit_cache_audit.json').read_text())
    if not gate['passed'] or not gate['projection_matches_pilot']:raise ValueError('Covariance cache audit failed')
    inventory,_,_,parent=parent_data()
    if receipt['inventory_sha256']!=parent['inventory_sha256']:raise ValueError('CSP inventory differs')
    rows=audit_cache(read_rows(PARENT/'metadata.csv'),np.load(PARENT/'vectors.npy',mmap_mode='r'),inventory)
    return rows,receipt


def training(rows):
    records=sorted([r for r in rows if r['role']=='fit' and r['variant'] in VARIANTS[:2]],
                   key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
    x,y,weights,sources=weighted_source_arrays(rows,NAMES,VARIANTS[:2],
        expected_source_counts={'rr':540,'chimera':720},seed=CONFIG['seed'])
    if len(x)!=7560 or not np.array_equal(x,[[float(r[n]) for n in NAMES] for r in records]):
        raise ValueError('CSP training traversal differs')
    return records,x,y,weights,sources


def fit(x,y,w,s,strength,manifest,wrong=None):
    return fit_csp_source(x,y,w,s,feature_names=NAMES,strength=strength,consistency_groups=wrong,
                          channels=32,filter_count=16,manifest_sha=manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'iteration.json').exists() or (OUTPUT/'crossfit.json').exists():
        raise FileExistsError('Preserve CSP experiment')
    control=json.loads((OUTPUT/'software_controls.json').read_text())
    if not control['passed'] or control['code_pins']!=code_pins():raise ValueError('CSP controls changed')
    rows,parent=inputs();manifest=parent['inventory_sha256'];records,x,truth,weights,sources=training(rows)
    if args.pilot:
        if (OUTPUT/'pilot.json').exists():raise FileExistsError('Preserve CSP pilot')
        folds=source_folds(records);selected=np.array([folds[r['domain'],r['src']]!=0 for r in records])
        _,groups=np.unique(sources[selected],return_inverse=True);start=perf_counter()
        with threadpool_limits(limits=1):
            rule,diagnostic=fit(x[selected],truth[selected],weights[selected],groups,10,manifest)
        elapsed=perf_counter()-start;path=OUTPUT/'pilot_fold_rule.json';rule.save(path)
        write_json(OUTPUT/'pilot.json',{'passed':elapsed<=120,'fit_s':elapsed,'projected_40_fit_s':40*elapsed,
            'rows':int(selected.sum()),'input_dimensions':2576,'mapped_dimensions':2064,'strength':10,
            'diagnostic':diagnostic,'code_pins':code_pins(),'features_receipt_sha256':file_sha256(PARENT/'features.json'),
            'rule_sha256':file_sha256(path),'assumption':'Same fold/count;conditional on strength-dependent solver iterations',
            'scope':'Actual train-fold CSP/head cost;no held or outer scoring'})
        print(json.dumps({'pilot_fit_s':elapsed,'projected40fits_s':40*elapsed}),flush=True)
        if elapsed>120:raise ValueError('CSP pilot too costly')
        return
    pilot=json.loads((OUTPUT/'pilot.json').read_text())
    if (not pilot['passed'] or pilot['code_pins']!=code_pins()
            or pilot['features_receipt_sha256']!=file_sha256(PARENT/'features.json')
            or pilot['rule_sha256']!=file_sha256(OUTPUT/'pilot_fold_rule.json')):
        raise ValueError('CSP pilot changed/failed')
    assignment,null=balanced_source_null(records,seed=CONFIG['seed'])
    prior=json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256']!=prior['assignment_sha256']:raise ValueError('CSP source null changed')
    sham=np.array([assignment[r['domain'],r['src']] for r in records]);zero=None

    def calibrate(rule,views):
        nonlocal zero
        fixed,diagnostic=calibrate_csp_rule(rule,views,class_threshold)
        if zero is None:zero=fixed
        return fixed,diagnostic

    def zero_control(margins):
        selected=[r for r in rows if r['role']=='selection']
        native=np.array([[float(r[n]) for n in NAMES] for r in selected])
        path=OUTPUT/'zero_portable_control.json';zero.save(path);restored=CSPRule.load(path)
        actual=zero.score(native);mapped=zero.readout.score(zero.mapper.transform(native))
        if (len(actual)!=5040 or not np.array_equal(actual,mapped)
                or not np.array_equal(actual,restored.score(native))
                or not np.array_equal(actual-zero.threshold,[r['score'] for r in margins])):
            raise ValueError('CSP zero portable scores differ')
        return {'records':5040,'portable_mapped_loaded_margin_exact':True,'scope':'Same-rule software identity only'}

    run_source_consistency(rows=rows,names=NAMES,records=records,x=x,truth=truth,sham=sham,
        weights=weights,sources=sources,strengths=STRENGTHS,variants=VARIANTS[:2],output=OUTPUT,
        fit=lambda a,b,c,d,e,f=None:fit(a,b,c,d,e,manifest,f),calibrate=calibrate,
        score=lambda a,b,c:score_rule(a,b,c,CONFIG),wrong_sources=wrong_sources(rows,sources),
        zero_reference=zero_control,null_intervals=null_intervals,pins=code_pins(),manifest_sha=manifest,
        assignment_sha=null['assignment_sha256'],features_sha=file_sha256(PARENT/'features.json'))


if __name__=='__main__':main()
