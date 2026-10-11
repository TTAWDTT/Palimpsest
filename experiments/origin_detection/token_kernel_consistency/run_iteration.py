"""Fit-local nonlinear kernel readout, followed by frozen source CV choice."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.kernel_source_consistency import (
    fit_kernel_source_consistency, calibrate_kernel_rule)
from palimpsest.detection.representations.frozen_cure_tokens import MIXED_NAMES
from palimpsest.evaluation.source_consistency_campaign import run_source_consistency
from palimpsest.evaluation.source_crossfit import source_folds
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.token_consistency_crossfit.run_iteration import (
    TOKENS, CONFIG, STRENGTHS, VARIANTS, inputs, training,
    code_pins as input_pins, class_threshold, score_rule, wrong_sources, null_intervals, write_json)

OUTPUT = WORK_DIR/'robust_statistics/token_kernel_consistency'


def code_pins():
    pins = input_pins()
    paths = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (
        'src/palimpsest/detection/algorithms/kernel_source_consistency.py',
        'src/palimpsest/detection/algorithms/kernel_readout.py',
        'src/palimpsest/evaluation/source_consistency_campaign.py',
        'tests/detection/test_kernel_source_consistency.py', 'tests/detection/test_kernel_readout.py',
        'tests/evaluation/test_source_consistency_campaign.py')]
    pins.update({str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)})
    return pins


def fit(x,y,w,s,strength,manifest,wrong=None):
    return fit_kernel_source_consistency(x,y,w,s,feature_names=MIXED_NAMES,strength=strength,
        consistency_groups=wrong,frequency_count=256,seed=20261006,manifest_sha=manifest)


def zero_control(margins):
    # Called after the new zero rule is saved by the calibration callback.
    from palimpsest.detection.algorithms.kernel_readout import KernelRule
    rule = KernelRule.load(OUTPUT/'zero_calibrated_control.json')
    rows,_ = inputs(); selected = [r for r in rows if r['role']=='selection']
    matrix = np.array([[float(r[n]) for n in MIXED_NAMES] for r in selected])
    mapped = np.c_[matrix,rule.mapper.transform(matrix)]
    native, direct = rule.score(matrix),rule.readout.score(mapped)
    if (len(native)!=5040 or not np.array_equal(native,direct)
            or not np.array_equal(native-rule.threshold,[r['score'] for r in margins])):
        raise ValueError('Nonlinear zero portable/mapped scores differ')
    return {'records':5040,'portable_mapped_margin_exact':True,
            'scope':'Same-rule mapping identity;not linear54 reference or independent science'}


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'iteration.json').exists() or (OUTPUT/'crossfit.json').exists():
        raise FileExistsError('Preserve token kernel experiment')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins']!=code_pins():raise ValueError('Kernel controls changed')
    rows,parent = inputs();manifest = parent['inventory_sha256']
    records,x,truth,weights,sources = training(rows)
    if args.pilot:
        if (OUTPUT/'pilot.json').exists():raise FileExistsError('Preserve kernel pilot')
        folds = source_folds(records);selected = np.array([folds[r['domain'],r['src']]!=0 for r in records])
        _,groups = np.unique(sources[selected],return_inverse=True);start = perf_counter()
        with threadpool_limits(limits=1):
            rule,diagnostic = fit(x[selected],truth[selected],weights[selected],groups,10,manifest)
        elapsed = perf_counter()-start;path = OUTPUT/'pilot_fold_rule.json';rule.save(path)
        write_json(OUTPUT/'pilot.json',{'passed':elapsed<=120,'fit_s':elapsed,'projected_40_fit_s':40*elapsed,
            'rows':int(selected.sum()),'input_dimensions':2048,'mapped_dimensions':2560,
            'strength':10,'diagnostic':diagnostic,'code_pins':code_pins(),
            'features_receipt_sha256':file_sha256(TOKENS/'features.json'),'rule_sha256':file_sha256(path),
            'assumption':'Same-fold dimensions/count;conditional on strength-dependent optimizer cost',
            'scope':'Real train-fold timing;no held or outer scores'})
        print(json.dumps({'pilot_fit_s':elapsed,'projected40fits_s':40*elapsed}),flush=True)
        if elapsed>120:raise ValueError('Kernel pilot too costly')
        return
    pilot = json.loads((OUTPUT/'pilot.json').read_text())
    if (not pilot['passed'] or pilot['code_pins']!=code_pins()
            or pilot['features_receipt_sha256']!=file_sha256(TOKENS/'features.json')
            or pilot['rule_sha256']!=file_sha256(OUTPUT/'pilot_fold_rule.json')):
        raise ValueError('Kernel pilot changed/failed')
    assignment,null = balanced_source_null(records,seed=CONFIG['seed'])
    expected = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256']!=expected['assignment_sha256']:raise ValueError('Kernel null mapping changed')
    sham = np.array([assignment[r['domain'],r['src']] for r in records]);calibration_count = 0

    def calibrate(rule,views):
        nonlocal calibration_count
        fixed,diagnostic = calibrate_kernel_rule(rule,views,class_threshold)
        if calibration_count==0:fixed.save(OUTPUT/'zero_calibrated_control.json')
        calibration_count += 1
        return fixed,diagnostic

    run_source_consistency(rows=rows,names=MIXED_NAMES,records=records,x=x,truth=truth,sham=sham,
        weights=weights,sources=sources,strengths=STRENGTHS,variants=VARIANTS[:2],output=OUTPUT,
        fit=lambda a,b,c,d,e,f=None:fit(a,b,c,d,e,manifest,f),calibrate=calibrate,
        score=lambda a,b,c:score_rule(a,b,c,CONFIG),wrong_sources=wrong_sources(rows,sources),
        zero_reference=zero_control,null_intervals=null_intervals,pins=code_pins(),manifest_sha=manifest,
        assignment_sha=null['assignment_sha256'],features_sha=file_sha256(TOKENS/'features.json'))


if __name__=='__main__':main()
