"""Preserve the fixed grid while allowing an unfinished solver more iterations."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.consistent_source_risk import fit_consistent_source_risk
from palimpsest.detection.representations.frozen_cure_tokens import MIXED_NAMES
from palimpsest.evaluation.source_consistency_campaign import run_source_consistency
from palimpsest.evaluation.source_crossfit import source_folds
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR, REPO_ROOT
from experiments.origin_detection.token_consistency_crossfit.run_iteration import (
    OUTPUT as PRIOR, TOKENS, CONFIG, STRENGTHS, VARIANTS, inputs, training, zero_reference,
    code_pins as prior_pins, class_threshold, score_rule, wrong_sources, null_intervals, write_json)

OUTPUT = WORK_DIR/'robust_statistics/token_consistency_solver'


def code_pins():
    pins = prior_pins()
    paths = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md',
        REPO_ROOT/'src/palimpsest/evaluation/source_consistency_campaign.py',
        REPO_ROOT/'tests/evaluation/test_source_consistency_campaign.py']
    pins.update({str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)})
    return pins


def prior_results():
    sealed = json.loads((PRIOR/'failure_receipt.json').read_text())
    controls = json.loads((PRIOR/'software_controls.json').read_text())
    expected = {'cv_truth_0.json','cv_truth_0.1.json','cv_truth_1.json'}
    if (sealed['status']!='incomplete' or sealed['outer_evaluation_executed']
            or sealed['code_pins']!=prior_pins() or controls['code_pins']!=prior_pins()
            or sealed['features_receipt_sha256']!=file_sha256(TOKENS/'features.json')
            or set(sealed['completed_cv_artifacts'])!=expected):
        raise ValueError('Prior partial protocol changed')
    values = {}
    for name, sha in sealed['completed_cv_artifacts'].items():
        path = PRIOR/name
        if file_sha256(path)!=sha:raise ValueError('Prior partial OOF changed')
        data = json.loads(path.read_text())
        if (len(data['scores'])!=7560 or len(data['fit_diagnostics'])!=5
                or any(d['maximum_absolute_gradient']>1e-5 or d['optimizer_iterations']>=500
                       or d['fit_records']!=6048 or d['sources']!=1008 for d in data['fit_diagnostics'])):
            raise ValueError('Prior partial fit diagnostics differ')
        values[name[len('cv_truth_'):-len('.json')]] = data
    return {'truth':values}, file_sha256(PRIOR/'failure_receipt.json')


def fit(x,y,w,s,strength,manifest,wrong=None):
    return fit_consistent_source_risk(x,y,w,s,strength=strength,consistency_groups=wrong,
        feature_names=MIXED_NAMES,temperature=.1,ridge=.01,scale_floor=.001,
        maximum_iterations=2000,gradient_tolerance=1e-5,manifest_sha=manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'iteration.json').exists() or (OUTPUT/'crossfit.json').exists():
        raise FileExistsError('Preserve solver continuation')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins']!=code_pins():raise ValueError('Solver controls changed')
    prior, failure_sha = prior_results()
    rows,parent = inputs();manifest = parent['inventory_sha256']
    records,x,truth,weights,sources = training(rows)
    if args.pilot:
        if (OUTPUT/'pilot.json').exists():raise FileExistsError('Preserve solver pilot')
        folds = source_folds(records);selected = np.array([folds[r['domain'],r['src']]!=0 for r in records])
        _,groups = np.unique(sources[selected],return_inverse=True);start = perf_counter()
        with threadpool_limits(limits=1):
            rule,diagnostic = fit(x[selected],truth[selected],weights[selected],groups,10,manifest)
        elapsed = perf_counter()-start;path = OUTPUT/'pilot_fold_rule.json';rule.save(path)
        write_json(OUTPUT/'pilot.json',{'passed':elapsed<=120,'fit_s':elapsed,'projected_25_fit_s':25*elapsed,
            'rows':int(selected.sum()),'dimensions':2048,'strength':10,'maximum_iterations':2000,
            'diagnostic':diagnostic,'code_pins':code_pins(),'prior_failure_sha256':failure_sha,
            'features_receipt_sha256':file_sha256(TOKENS/'features.json'),'rule_sha256':file_sha256(path),
            'assumption':'Same train-fold dimensions;linear count;strength-dependent solver cost',
            'scope':'Real train-fold cost;held scores and outer roles not evaluated'})
        print(json.dumps({'pilot_fit_s':elapsed,'projected25fits_s':25*elapsed,'iterations':diagnostic['optimizer_iterations']}),flush=True)
        if elapsed>120:raise ValueError('Continuation pilot too costly')
        return
    pilot = json.loads((OUTPUT/'pilot.json').read_text())
    if (not pilot['passed'] or pilot['code_pins']!=code_pins() or pilot['prior_failure_sha256']!=failure_sha
            or pilot['features_receipt_sha256']!=file_sha256(TOKENS/'features.json')
            or pilot['rule_sha256']!=file_sha256(OUTPUT/'pilot_fold_rule.json')):
        raise ValueError('Continuation pilot changed/failed')
    assignment,null = balanced_source_null(records,seed=CONFIG['seed'])
    expected = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256']!=expected['assignment_sha256']:raise ValueError('Null mapping changed')
    sham = np.array([assignment[r['domain'],r['src']] for r in records])
    run_source_consistency(rows=rows,names=MIXED_NAMES,records=records,x=x,truth=truth,sham=sham,
        weights=weights,sources=sources,strengths=STRENGTHS,variants=VARIANTS[:2],output=OUTPUT,
        fit=lambda a,b,c,d,e,f=None:fit(a,b,c,d,e,manifest,f),calibrate=class_threshold,
        score=lambda a,b,c:score_rule(a,b,c,CONFIG),wrong_sources=wrong_sources(rows,sources),
        zero_reference=zero_reference,null_intervals=null_intervals,pins=code_pins(),manifest_sha=manifest,
        assignment_sha=null['assignment_sha256'],features_sha=file_sha256(TOKENS/'features.json'),prior=prior)


if __name__=='__main__':main()
