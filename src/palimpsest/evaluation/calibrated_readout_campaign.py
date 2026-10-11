"""Shared driver for the frozen 1260/420/420-source readout development protocol.

Method and dataset bindings are explicit. This driver keeps controls, measured
pilot, fit/cal/held CV and four outer evaluations in order, refusing overwrite.
It does not download data, run image encoders, or change source roles.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from .balanced_null import balanced_source_null
from .calibrated_source_crossfit import calibrated_crossfit
from .source_consistency_campaign import run_source_consistency
from .source_crossfit import source_folds
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR


@dataclass(frozen=True)
class CampaignData:
    names: tuple
    variants: tuple
    seed: int
    features_receipt: Path
    inputs: Callable
    training: Callable
    calibration_views: Callable
    wrong_sources: Callable
    score: Callable
    null_intervals: Callable


@dataclass(frozen=True)
class CampaignMethod:
    parameters: tuple
    pilot_parameters: tuple
    tests: tuple
    pins: Callable
    fitter: Callable
    calibrate: Callable
    load: Callable


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def run_calibrated_campaign(output, args, data, method):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'iteration.json').exists() or (output/'crossfit.json').exists():
        raise FileExistsError('Preserve calibrated readout calculation')
    if args.record_controls:
        if (output/'software_controls.json').exists(): raise FileExistsError('Preserve controls')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'pytest', '-q', *method.tests],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding='utf-8')
        print(result.stdout, flush=True)
        if result.returncode or ' passed' not in result.stdout or 'skipped' in result.stdout:
            raise ValueError('Readout controls failed: '+result.stderr)
        write_json(output/'software_controls.json', {'passed': True, 'stdout': result.stdout,
            'tests': method.tests, 'code_pins': method.pins(), 'scope': 'Software plants;not independent image evidence'})
        return
    controls = json.loads((output/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins'] != method.pins(): raise ValueError('Controls changed')
    rows, parent = data.inputs(); manifest = parent['inventory_sha256']
    records, x, truth, weights, sources = data.training(rows)
    fitter = method.fitter(manifest); folds = source_folds(records)
    ids = np.array([folds[r['domain'], r['src']] for r in records])
    if args.pilot:
        if (output/'pilot.json').exists(): raise FileExistsError('Preserve pilot')
        train = (ids != 0) & (ids != 1); cal = ids == 1; held = ids == 0
        _, groups = np.unique(sources[train], return_inverse=True)
        views = data.calibration_views(records, x, truth, cal); times = []; diagnostics = []; rule = None
        with threadpool_limits(limits=1):
            for parameter in method.pilot_parameters:
                start = perf_counter()
                rule, d = fitter.fit(x[train], truth[train], weights[train], groups, parameter)
                rule, d['inner_calibration'] = method.calibrate(rule, views)
                values = rule.score(x[held])-rule.threshold
                if len(values) != 1512 or not np.isfinite(values).all(): raise ValueError('Pilot margin invalid')
                times.append(perf_counter()-start); diagnostics.append(d)
        if fitter.audit()['banks'] != 1 or len(times) != 2: raise ValueError('Pilot cache or timing count differs')
        path = output/'pilot_fold_rule.json'; rule.save(path)
        write_json(output/'pilot.json', {'passed': max(times) <= 120, 'cold_s': times[0], 'warm_s': times[1],
            'projected40cv_s': 10*times[0]+30*times[1], 'diagnostics': diagnostics, 'code_pins': method.pins(),
            'features_receipt_sha256': file_sha256(data.features_receipt), 'rule_sha256': file_sha256(path),
            'scope': 'Actual three-fit-fold plus cal/held;not outer or independent evaluation',
            'assumption': '10 cold+30 warm;other parameters may cost differently;input/fullfit/outer excluded'})
        print(json.dumps({'cold_s': times[0], 'warm_s': times[1], 'projected40cv_s': 10*times[0]+30*times[1]}), flush=True)
        if max(times) > 120: raise ValueError('Pilot too costly')
        return
    pilot = json.loads((output/'pilot.json').read_text())
    if (not pilot['passed'] or pilot['code_pins'] != method.pins()
            or pilot['features_receipt_sha256'] != file_sha256(data.features_receipt)
            or pilot['rule_sha256'] != file_sha256(output/'pilot_fold_rule.json')):
        raise ValueError('Pilot changed/failed')
    assignment, null = balanced_source_null(records, seed=data.seed)
    old = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256'] != old['assignment_sha256']: raise ValueError('Null assignment changed')
    sham = np.array([assignment[r['domain'], r['src']] for r in records]); cv = {}; start = perf_counter()
    with threadpool_limits(limits=1):
        for mode, labels in (('truth', truth), ('null', sham)):
            def progress(parameter, result):
                write_json(output/f'cv_{mode}_{parameter}.json', result)
                print(json.dumps({'cv': mode, 'parameter': parameter,
                    'minimum_ba': result['minimum_all_scope_ba'], 'maximum_drop': result['maximum_all_scope_drop']}), flush=True)
            cv[mode], _, actual = calibrated_crossfit(records, x, labels, weights, sources,
                method.parameters, data.variants[:2], data.names, fitter.fit, method.calibrate, progress)
            if actual != folds: raise ValueError('Source folds changed')
    cv_elapsed = perf_counter()-start
    write_json(output/'inner_roles.json', {'fold_sources': {'/'.join(k): v for k, v in folds.items()},
        'calibration_fold_rule': '(held+1)%5', 'train_sources': 756, 'cal_sources': 252,
        'held_sources': 252, 'cv_elapsed_s': cv_elapsed, 'code_pins': method.pins()})
    zero = None

    def calibrate(rule, views):
        nonlocal zero
        fixed, diagnostic = method.calibrate(rule, views)
        if zero is None: zero = fixed
        return fixed, diagnostic

    def zero_control(margins):
        native = np.array([[float(r[n]) for n in data.names] for r in rows if r['role'] == 'selection'])
        path = output/'zero_portable_control.json'; zero.save(path); restored = method.load(path)
        values = zero.score(native)
        if (len(values) != 5040 or not np.array_equal(values, zero.readout.score(zero.transform(native)))
                or not np.array_equal(values, restored.score(native))
                or not np.array_equal(values-zero.threshold, [r['score'] for r in margins])):
            raise ValueError('Portable mapped/native/loaded margin identity differs')
        return {'records': 5040, 'portable_mapped_loaded_margin_exact': True, 'scope': 'Same-rule software identity'}

    run_source_consistency(rows=rows, names=data.names, records=records, x=x, truth=truth, sham=sham,
        weights=weights, sources=sources, strengths=method.parameters, variants=data.variants[:2], output=output,
        fit=fitter.fit, calibrate=calibrate, score=data.score, wrong_sources=data.wrong_sources(rows, sources),
        zero_reference=zero_control, null_intervals=data.null_intervals, pins=method.pins(), manifest_sha=manifest,
        assignment_sha=null['assignment_sha256'], features_sha=file_sha256(data.features_receipt), prior=cv)
    audit = fitter.audit()
    if audit['banks'] != 12: raise ValueError('Expected ten inner plus two fullfit banks')
    write_json(output/'bank_cache_audit.json', {**audit, 'cv_lp_fits': 40, 'final_lp_fits': 4})
