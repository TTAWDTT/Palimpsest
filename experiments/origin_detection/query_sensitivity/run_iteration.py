"""Fixed-head native-condition experiment on two cached sensitivity values."""

import argparse
from collections import Counter
import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.robust_views import evaluate_views
from palimpsest.evaluation.source_crossfit import source_folds
from palimpsest.evaluation.source_panel_crossfit import wrong_panel_sources
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from experiments.origin_detection.query_sensitivity.protocol import OUTPUT, NAMES, SEED, code_pins, write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold


def views(rows):
    return {key: value for processed in (False, True)
        for key, value in feature_views(rows, NAMES, 'threshold', processed=processed).items()}


def controls():
    import subprocess
    import sys
    tests = ('tests/detection/test_query_sensitivity.py', 'tests/detection/test_source_view_risk.py')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stdout+result.stderr)
    write_json(OUTPUT/'software_controls.json', {'passed': True, 'stdout': result.stdout,
        'tests': tests, 'code_pins': code_pins(), 'scope': 'Component controls,not full pipeline certification'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--record-controls', action='store_true')
    group.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.record_controls:
        if (OUTPUT/'software_controls.json').exists():
            raise FileExistsError('Preserve controls')
        controls()
        return
    control = json.loads((OUTPUT/'software_controls.json').read_text())
    receipt = json.loads((OUTPUT/'features.json').read_text())
    if (not control['passed'] or control['code_pins'] != code_pins()
            or not receipt['passed'] or receipt['code_pins'] != code_pins()
            or receipt['csv_sha256'] != file_sha256(OUTPUT/'features.csv')):
        raise ValueError('Sensitivity inputs/controls changed')
    rows = read_rows(OUTPUT/'features.csv')
    records = sorted([r for r in rows if r['role'] == 'fit'],
        key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    x, labels, weights, sources = weighted_source_arrays(rows, NAMES, ('raw',),
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=SEED)
    if Counter(r['role'] for r in rows) != {'fit': 3780, 'threshold': 1260, 'selection': 1260}:
        raise ValueError('Frozen role counts differ')
    roles = {role: {(r['domain'], r['src']) for r in rows if r['role'] == role}
        for role in ('fit', 'threshold', 'selection')}
    if any(roles[a]&roles[b] for a, b in (('fit', 'threshold'), ('fit', 'selection'), ('threshold', 'selection'))):
        raise ValueError('Source role leakage')
    if args.pilot:
        if (OUTPUT/'pilot.json').exists():
            raise FileExistsError('Preserve fit pilot')
        folds = source_folds(records)
        ids = np.array([folds[r['domain'], r['src']] for r in records])
        mask = (ids != 0)&(ids != 1)
        _, groups = np.unique(sources[mask], return_inverse=True)
        start = perf_counter()
        with threadpool_limits(limits=1):
            rule, diagnostic = fit_source_risk(x[mask], labels[mask], weights[mask], groups,
                feature_names=NAMES, manifest_sha=receipt['inventory_sha256'])
            margins = rule.score(x[ids == 0])
        elapsed = perf_counter()-start
        if len(margins) != 756 or not np.isfinite(margins).all():
            raise ValueError('Pilot held scores invalid')
        write_json(OUTPUT/'pilot.json', {'passed': elapsed < 120, 'fit_score_s': elapsed,
            'train_rows': int(mask.sum()), 'held_rows': len(margins), 'diagnostic': diagnostic,
            'code_pins': code_pins(), 'feature_receipt_sha256': file_sha256(OUTPUT/'features.json'),
            'scope': 'Fixed-head fit-role pilot;fullfit/calibration/bootstrap/loading excluded',
            'projection': 'Four full fits require measurement at larger row count;not a task ETA'})
        print({'pilot_s': elapsed}, flush=True)
        return
    if (OUTPUT/'iteration.json').exists():
        raise FileExistsError('Preserve sensitivity scores')
    pilot = json.loads((OUTPUT/'pilot.json').read_text())
    if not pilot['passed'] or pilot['code_pins'] != code_pins():
        raise ValueError('Fit pilot changed')
    assignment, null = balanced_source_null(records, seed=SEED)
    sham = np.array([assignment[r['domain'], r['src']] for r in records])
    wrong = wrong_panel_sources(records, sources, seed=SEED)
    specifications = (('mean', labels, sources, 0), ('source', labels, sources, .1),
                      ('wrong_source', labels, wrong, .1), ('null', sham, sources, .1))
    selected = [r for r in rows if r['role'] == 'selection']
    matrix = np.array([[float(r[name]) for name in NAMES] for r in selected])
    scored, candidates, diagnostics = {}, {}, {}
    fit_times = {}
    start = perf_counter()
    with threadpool_limits(limits=1):
        for name, truth, groups, temperature in specifications:
            tick = perf_counter()
            rule, diagnostic = fit_source_risk(x, truth, weights, groups, feature_names=NAMES,
                temperature=temperature, manifest_sha=receipt['inventory_sha256'])
            fit_times[name] = perf_counter()-tick
            rule, calibration = class_threshold(rule, views(rows))
            path = OUTPUT/f'{name}_rule.json'
            rule.save(path)
            loaded = StableRule.load(path)
            scores = rule.score(matrix)-rule.threshold
            if (not np.array_equal(scores, loaded.score(matrix)-loaded.threshold)
                    or not np.array_equal(scores, [rule.score([v])[0]-rule.threshold for v in matrix])):
                raise ValueError('Saved/single score parity failed')
            scored[name] = [{**{key: r[key] for key in ('filename', 'domain', 'scene', 'condition',
                'variant', 'src', 'role', 'label')}, 'score': float(score)} for r, score in zip(selected, scores)]
            result = evaluate_views(scored[name], variant_order=('raw',))
            if len(result['metrics']) != 15 or len(result['pairs']) != 10:
                raise ValueError('Native evaluation panel changed')
            candidates[name] = result
            diagnostics[name] = {**diagnostic, 'calibration': calibration, 'rule_sha256': file_sha256(path)}
            print({'head': name, 'minimum_ba': result['minimum_domain_ba'],
                   'maximum_drop': result['maximum_any_scene_drop']}, flush=True)
    write_json(OUTPUT/'selection_scores.json', scored)
    write_json(OUTPUT/'iteration.json', {'candidates': candidates, 'fit_diagnostics': diagnostics,
        'fit_s': fit_times, 'elapsed_s': perf_counter()-start, 'code_pins': code_pins(),
        'feature_receipt_sha256': file_sha256(OUTPUT/'features.json'),
        'selection_scores_sha256': file_sha256(OUTPUT/'selection_scores.json'),
        'null_assignment': null, 'role_sources': {k: len(v) for k, v in roles.items()},
        'scope': 'Native conditions only;no extra encoded query robustness or independent evidence',
        'goal_achieved': False})


if __name__ == '__main__':
    main()
