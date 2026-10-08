"""Whole-pipeline calibrated source CV for compact maximum-hinge readout."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.source_hinge import MarginFitRefused
from palimpsest.detection.algorithms.source_score_margin import ScoreMarginFitter
from palimpsest.detection.algorithms.source_score_subspace import ScoreSubspaceRule, calibrate_score_subspace
from palimpsest.evaluation.calibrated_source_crossfit import calibrated_crossfit
from palimpsest.evaluation.source_consistency_campaign import run_source_consistency
from palimpsest.evaluation.source_crossfit import source_folds
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR, REPO_ROOT
from experiments.origin_detection.calibrated_score_subspace.run_iteration import (
    PARENT, NAMES, CONFIG, VARIANTS, inputs, training, inner_views,
    code_pins as input_pins, class_threshold, score_rule, wrong_sources, null_intervals, write_json)

OUTPUT = WORK_DIR/'robust_statistics/source_score_margin'
STRENGTHS = (.0001, .001, .01, .1)
TESTS = ('tests/detection/test_source_score_margin.py', 'tests/detection/test_source_hinge.py',
         'tests/detection/test_source_score_subspace.py', 'tests/evaluation/test_calibrated_source_crossfit.py')


def code_pins():
    pins = input_pins()
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (*TESTS, 'src/palimpsest/detection/algorithms/source_score_margin.py',
                                 'src/palimpsest/detection/algorithms/source_hinge.py')]
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)})
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--record-controls', action='store_true')
    args = parser.parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT/'iteration.json').exists() or (OUTPUT/'crossfit.json').exists():
        raise FileExistsError('Preserve compact margin experiment')
    if args.record_controls:
        if (OUTPUT/'software_controls.json').exists(): raise FileExistsError('Preserve controls')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'pytest', '-q', *TESTS],
                                cwd=REPO_ROOT, capture_output=True, text=True, encoding='utf-8')
        print(result.stdout, flush=True)
        if result.returncode or ' passed' not in result.stdout or 'skipped' in result.stdout:
            raise ValueError('Compact margin controls failed: '+result.stderr)
        write_json(OUTPUT/'software_controls.json', {'passed': True, 'stdout': result.stdout,
            'tests': TESTS, 'code_pins': code_pins(), 'scope': 'Constructed software truth;not independent image science'})
        return
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins'] != code_pins(): raise ValueError('Margin controls changed')
    rows, parent = inputs(); manifest = parent['inventory_sha256']
    records, x, truth, weights, sources = training(rows)
    fitter = ScoreMarginFitter(NAMES, manifest_sha=manifest)
    folds = source_folds(records); ids = np.array([folds[r['domain'], r['src']] for r in records])
    if args.pilot:
        if (OUTPUT/'pilot.json').exists(): raise FileExistsError('Preserve margin pilot')
        train = (ids != 0) & (ids != 1); cal = ids == 1; held = ids == 0
        _, groups = np.unique(sources[train], return_inverse=True)
        views = inner_views(records, x, truth, cal); times = []; diagnostics = []; rule = None
        with threadpool_limits(limits=1):
            for penalty in (STRENGTHS[0], STRENGTHS[-1]):
                start = perf_counter()
                rule, d = fitter.fit(x[train], truth[train], weights[train], groups, penalty)
                rule, d['inner_calibration'] = calibrate_score_subspace(rule, views, class_threshold)
                values = rule.score(x[held])-rule.threshold
                if len(values) != 1512 or not np.isfinite(values).all(): raise ValueError('Pilot held margin invalid')
                times.append(perf_counter()-start); diagnostics.append(d)
        if fitter.provider.builds != 1: raise ValueError('Margin pilot bank rebuilt')
        path = OUTPUT/'pilot_fold_rule.json'; rule.save(path)
        write_json(OUTPUT/'pilot.json', {'passed': max(times) <= 120, 'cold_s': times[0], 'warm_s': times[1],
            'projected40cv_s': 10*times[0]+30*times[1], 'diagnostics': diagnostics, 'code_pins': code_pins(),
            'features_receipt_sha256': file_sha256(PARENT/'features.json'), 'rule_sha256': file_sha256(path),
            'scope': 'Same-machine three-fit-fold plus calibration/held timing;not outer or independent validation',
            'assumption': '10 cold+30 warm;other penalties and iteration counts may differ;loading/fullfit excluded'})
        print(json.dumps({'cold_s': times[0], 'warm_s': times[1], 'projected40cv_s': 10*times[0]+30*times[1]}), flush=True)
        if max(times) > 120: raise ValueError('Margin pilot too costly')
        return
    pilot = json.loads((OUTPUT/'pilot.json').read_text())
    if (not pilot['passed'] or pilot['code_pins'] != code_pins()
            or pilot['features_receipt_sha256'] != file_sha256(PARENT/'features.json')
            or pilot['rule_sha256'] != file_sha256(OUTPUT/'pilot_fold_rule.json')):
        raise ValueError('Margin pilot changed/failed')
    assignment, null = balanced_source_null(records, seed=CONFIG['seed'])
    old = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256'] != old['assignment_sha256']: raise ValueError('Null assignment changed')
    sham = np.array([assignment[r['domain'], r['src']] for r in records]); cv = {}; start = perf_counter()
    with threadpool_limits(limits=1):
        for mode, labels in (('truth', truth), ('null', sham)):
            def progress(penalty, result):
                write_json(OUTPUT/f'cv_{mode}_{penalty}.json', result)
                print(json.dumps({'cv': mode, 'penalty': penalty, 'minimum_ba': result['minimum_all_scope_ba'],
                    'maximum_drop': result['maximum_all_scope_drop']}), flush=True)
            cv[mode], _, actual = calibrated_crossfit(records, x, labels, weights, sources,
                STRENGTHS, VARIANTS[:2], NAMES, fitter.fit,
                lambda rule, views: calibrate_score_subspace(rule, views, class_threshold), progress)
            if actual != folds: raise ValueError('Margin source folds changed')
    cv_elapsed = perf_counter()-start
    write_json(OUTPUT/'inner_roles.json', {'fold_sources': {'/'.join(k): v for k, v in folds.items()},
        'calibration_fold_rule': '(held+1)%5', 'train_sources': 756, 'cal_sources': 252,
        'held_sources': 252, 'cv_elapsed_s': cv_elapsed, 'code_pins': code_pins()})
    zero = None

    def calibrate(rule, views):
        nonlocal zero
        fixed, diagnostic = calibrate_score_subspace(rule, views, class_threshold)
        if zero is None: zero = fixed
        return fixed, diagnostic

    def zero_control(margins):
        selected = [r for r in rows if r['role'] == 'selection']
        native = np.array([[float(r[n]) for n in NAMES] for r in selected])
        path = OUTPUT/'zero_portable_control.json'; zero.save(path)
        restored = ScoreSubspaceRule.load(path); values = zero.score(native)
        if (len(values) != 5040 or not np.array_equal(values, zero.readout.score(zero.bank.transform(native)))
                or not np.array_equal(values, restored.score(native))
                or not np.array_equal(values-zero.threshold, [r['score'] for r in margins])):
            raise ValueError('Margin portable identity differs')
        return {'records': 5040, 'portable_mapped_loaded_margin_exact': True, 'scope': 'Shared-rule software identity'}

    run_source_consistency(rows=rows, names=NAMES, records=records, x=x, truth=truth, sham=sham,
        weights=weights, sources=sources, strengths=STRENGTHS, variants=VARIANTS[:2], output=OUTPUT,
        fit=fitter.fit, calibrate=calibrate, score=lambda a, b, c: score_rule(a, b, c, CONFIG),
        wrong_sources=wrong_sources(rows, sources), zero_reference=zero_control, null_intervals=null_intervals,
        pins=code_pins(), manifest_sha=manifest, assignment_sha=null['assignment_sha256'],
        features_sha=file_sha256(PARENT/'features.json'), prior=cv)
    if fitter.provider.builds != 12: raise ValueError('Margin bank count differs')
    write_json(OUTPUT/'bank_cache_audit.json', {'passed': True, 'banks': 12, 'base_fits': 36,
        'auxiliary_logistic_head_fits': 12, 'cv_lp_fits': 40, 'final_lp_fits': 4,
        'training_array_keys': sorted(fitter.provider.bases), 'scope': 'Execution counts;not data independence'})


if __name__ == '__main__':
    try:
        main()
    except MarginFitRefused as error:
        write_json(OUTPUT/'failure_receipt.json', {'passed': False, 'diagnostic': error.diagnostic,
            'code_pins': code_pins(), 'scope': 'Refused LP execution;not detector failure evidence'})
        raise
