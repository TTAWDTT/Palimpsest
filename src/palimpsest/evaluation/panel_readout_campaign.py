"""Readout driver for an explicitly expanded source-view panel.

Reuses immutable control recording, algorithms, portable checks and evaluators.
Panel cardinalities replace the old six-view budgets without silent bypasses.
"""

from dataclasses import asdict
import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .calibrated_readout_campaign import run_calibrated_campaign, write_json
from .source_panel_crossfit import panel_masks, calibrated_panel_crossfit
from .source_crossfit import choose_strength
from .balanced_null import balanced_source_null
from .features import feature_views
from .training_fit import fit_training_rows
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR


def run_panel_campaign(output, args, data, method, panel):
    if args.record_controls:
        return run_calibrated_campaign(output, args, data, method)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'iteration.json').exists() or (output/'crossfit.json').exists(): raise FileExistsError('Preserve panel run')
    controls = json.loads((output/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins'] != method.pins(): raise ValueError('Panel controls changed')
    rows, parent = data.inputs(); manifest = parent['inventory_sha256']
    records, x, truth, weights, sources = data.training(rows)
    train, cal, held, _, folds = panel_masks(records, sources, panel)
    fitter = method.fitter(manifest)
    record_aware = getattr(method, 'record_aware', False)
    if args.pilot:
        if (output/'pilot.json').exists(): raise FileExistsError('Preserve panel pilot')
        _, groups = np.unique(sources[train], return_inverse=True)
        views = data.calibration_views(records, x, truth, cal); times = []; diagnostics = []; rule = None
        with threadpool_limits(limits=1):
            for parameter in method.pilot_parameters:
                start = perf_counter()
                context = tuple(r for r, flag in zip(records, train) if flag)
                rule, d = fit_training_rows(fitter.fit, x[train], truth[train], weights[train], groups, parameter,
                    records=context, record_aware=record_aware)
                rule, d['inner_calibration'] = method.calibrate(rule, views)
                values = rule.score(x[held])-rule.threshold
                if values.shape != (int(held.sum()),) or not np.isfinite(values).all(): raise ValueError('Pilot held margins')
                times.append(perf_counter()-start); diagnostics.append(d)
        if len(times) != 2 or fitter.audit()['banks'] != 1: raise ValueError('Panel pilot cache/times')
        path = output/'pilot_fold_rule.json'; rule.save(path)
        write_json(output/'pilot.json', {'passed': max(times) <= 120, 'cold_s': times[0], 'warm_s': times[1],
            'projected40cv_s': 10*times[0]+30*times[1], 'diagnostics': diagnostics, 'source_panel': asdict(panel),
            'code_pins': method.pins(), 'features_receipt_sha256': file_sha256(data.features_receipt),
            'rule_sha256': file_sha256(path), 'scope': 'Panel train/cal/held cost;not independent or outer validation',
            'assumption': '10 cold+30 warm,other parameters vary;input/fullfit/outer excluded'})
        print(json.dumps({'cold_s': times[0], 'warm_s': times[1], 'projected40cv_s': 10*times[0]+30*times[1]}), flush=True)
        if max(times) > 120: raise ValueError('Panel cost pilot too large')
        return
    pilot = json.loads((output/'pilot.json').read_text())
    if (not pilot['passed'] or pilot['code_pins'] != method.pins() or pilot['source_panel'] != json.loads(json.dumps(asdict(panel)))
            or pilot['features_receipt_sha256'] != file_sha256(data.features_receipt)
            or pilot['rule_sha256'] != file_sha256(output/'pilot_fold_rule.json')):
        raise ValueError('Panel pilot changed/failed')
    assignment, null = balanced_source_null(records, seed=data.seed)
    old = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256'] != old['assignment_sha256']: raise ValueError('Panel null mapping changed')
    sham = np.array([assignment[r['domain'], r['src']] for r in records]); cv = {}; start = perf_counter()
    with threadpool_limits(limits=1):
        for mode, labels in (('truth', truth), ('null', sham)):
            def progress(parameter, result):
                write_json(output/f'cv_{mode}_{parameter}.json', result)
                print(json.dumps({'cv': mode, 'parameter': parameter, 'minimum_ba': result['minimum_all_scope_ba'],
                                  'maximum_drop': result['maximum_all_scope_drop']}), flush=True)
            cv[mode], _, actual = calibrated_panel_crossfit(records, x, labels, weights, sources,
                method.parameters, data.names, panel, fitter.fit, method.calibrate, progress,
                record_aware=record_aware)
            if actual != folds: raise ValueError('Panel source folds changed')
    cv_elapsed = perf_counter()-start; chosen = {mode: choose_strength(candidates) for mode, candidates in cv.items()}
    write_json(output/'crossfit.json', {'results': cv, 'chosen': chosen, 'fold_seed': 20261007,
        'fold_sources': {'/'.join(k): v for k, v in folds.items()}, 'source_panel': asdict(panel),
        'code_pins': method.pins(), 'inventory_sha256': manifest, 'scope': 'Expanded fit-only calibrated OOF;not external'})
    write_json(output/'inner_roles.json', {'cv_elapsed_s': cv_elapsed, 'source_panel': asdict(panel),
        'train_sources': panel.sources*(panel.folds-2)//panel.folds,
        'cal_sources': panel.sources//panel.folds, 'held_sources': panel.sources//panel.folds})
    views = {key+'/'+v: view for v in panel.variants for processed in (False, True)
             for key, view in feature_views(rows, data.names, 'threshold', processed=processed, variant=v).items()}
    if len(views) != panel.calibration_groups_per_variant*len(panel.variants): raise ValueError('Final panel cal coverage')
    results = {}; scores = {}; artifacts = {}; reference = None; start = perf_counter()
    with threadpool_limits(limits=1):
        for key, labels, parameter, wrong in (
            ('zero', truth, 0, None), ('selected', truth, float(chosen['truth'][0]), None),
            ('selected_wrong_source', truth, float(chosen['truth'][0]), data.wrong_sources(rows, sources)),
            ('selected_null', sham, float(chosen['null'][0]), None)):
            rule, d = fit_training_rows(fitter.fit, x, labels, weights, sources, parameter,
                records=records, record_aware=record_aware, wrong=wrong)
            rule, d['calibration'] = method.calibrate(rule, views)
            margins, result = data.score(rows, data.names, rule)
            if key == 'zero':
                native = np.array([[float(r[n]) for n in data.names] for r in rows if r['role'] == 'selection'])
                path = output/'zero_portable_control.json'; rule.save(path); restored = method.load(path); actual = rule.score(native)
                if (len(actual) != 5040 or not np.array_equal(actual, restored.score(native))
                        or not np.array_equal(actual, rule.readout.score(rule.transform(native)))
                        or not np.array_equal(actual-rule.threshold, [r['score'] for r in margins])):
                    raise ValueError('Panel portable identity differs')
                reference = {'records': 5040, 'portable_mapped_loaded_margin_exact': True, 'scope': 'Software only'}
            result.update({'fit_diagnostics': d, 'threshold': rule.threshold, 'chosen_strength': parameter})
            path = output/(key+'_rule.json'); rule.save(path)
            results[key], scores[key], artifacts[key] = result, margins, file_sha256(path)
            print(json.dumps({'candidate': key, 'parameter': parameter, 'minimum_ba': result['minimum_domain_ba'],
                'minimum_scene_ba': result['minimum_scene_ba'], 'maximum_drop': result['maximum_any_scene_drop']}), flush=True)
    write_json(output/'selection_scores.json', scores)
    write_json(output/'iteration.json', {'candidates': results, 'chosen_strengths': chosen, 'source_panel': asdict(panel),
        'code_pins': method.pins(), 'features_receipt_sha256': file_sha256(data.features_receipt),
        'inventory_sha256': manifest, 'balanced_assignment_sha256': null['assignment_sha256'],
        'zero_reference_gate': reference, 'cv_elapsed_s': cv_elapsed, 'elapsed_s': perf_counter()-start,
        'crossfit_sha256': file_sha256(output/'crossfit.json'), 'selection_scores_sha256': file_sha256(output/'selection_scores.json'),
        'null_raw_processed_auc_ci95': data.null_intervals(scores['selected_null']), 'rule_files': artifacts,
        'goal_achieved': False, 'scope': 'Repeated development;Q60 now fit/cal support;not unseen-Q60 or independent validation'})
    audit = fitter.audit()
    if audit['banks'] != 12: raise ValueError('Panel bank count differs')
    write_json(output/'bank_cache_audit.json', {**audit,
        'cv_readout_fits': 2*panel.folds*len(method.parameters), 'final_readout_fits': 4})
