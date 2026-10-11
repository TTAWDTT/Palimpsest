"""Fixed stronger support and conventional full CuRe source-risk readouts."""

import json
from time import perf_counter

from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
from palimpsest.detection.representations.frozen_cure import FEATURE_NAMES, FULL_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views, validate_feature_cache
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs, null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS
from experiments.origin_detection.cure_encoding_risk.prepare_features import OUTPUT, code_pins
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold

CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def main():
    if (OUTPUT / 'iteration.json').exists():
        raise FileExistsError('Preserve stronger CuRe risk campaign')
    rows, parent = cure_inputs()
    receipt = json.loads((OUTPUT / 'features.json').read_text())
    if (receipt['code_pins'] != code_pins() or receipt['csv_sha256'] != file_sha256(OUTPUT / 'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256'] or receipt['records'] != 5040):
        raise ValueError('Additional support evidence changed')
    old, inventory, _, _ = parent_data()
    del old
    additional = read_rows(OUTPUT / 'features.csv')
    validate_feature_cache(additional, [r for r in inventory if r['role'] != 'selection'], FEATURE_NAMES,
        variants=(VARIANTS[2],), bounds=(-float('inf'), float('inf')))
    rows += additional
    if len(rows) != 20160:
        raise ValueError('Joined stronger support coverage changed')
    for role in ('fit', 'threshold', 'selection'):
        validate_feature_cache([r for r in rows if r['role'] == role], [r for r in inventory if r['role'] == role],
            FEATURE_NAMES, variants=VARIANTS if role == 'selection' else VARIANTS[:3],
            bounds=(-float('inf'), float('inf')))
    assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Balanced source assignment changed')
    results, rules, scores = {}, {}, {}
    start = perf_counter()
    with threadpool_limits(limits=1):
        for key, temperature in [('source', .1), ('mean', 0), ('source_null', .1)]:
            selected = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
                        if r['role'] == 'fit' else r for r in rows] if key.endswith('_null') else rows
            x, y, weights, sources = weighted_source_arrays(selected, FULL_NAMES, VARIANTS[:3],
                expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
            if len(x) != 11340:
                raise ValueError('Strong fit count changed')
            rule, diagnostic = fit_source_risk(x, y, weights, sources, feature_names=FULL_NAMES,
                temperature=temperature, ridge=.01, scale_floor=.001, maximum_iterations=500,
                gradient_tolerance=1e-5, manifest_sha=parent['inventory_sha256'])
            views = {k + '/' + v: view for v in VARIANTS[:3] for p in (False, True)
                for k, view in feature_views(rows, FULL_NAMES, 'threshold', processed=p, variant=v).items()}
            if len(views) != 36:
                raise ValueError('Strong threshold groups changed')
            rule, diagnostic['calibration'] = class_threshold(rule, views)
            margins, result = score_rule(rows, FULL_NAMES, rule, CONFIG)
            result.update({'fit_diagnostics': diagnostic, 'threshold': rule.threshold})
            results[key], rules[key], scores[key] = result, rule, margins
            print(json.dumps({'candidate': key, 'minimum_ba': result['minimum_domain_ba'],
                'minimum_scene_ba': result['minimum_scene_ba'],
                'maximum_scene_drop': result['maximum_any_scene_drop']}), flush=True)
    for k, rule in rules.items():
        rule.save(OUTPUT / (k+'_rule.json'))
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(OUTPUT / 'iteration.json', {'candidates': results, 'code_pins': code_pins(),
        'additional_features_sha256': receipt['csv_sha256'],
        'null_raw_processed_auc_ci95': null_intervals(scores['source_null']), 'elapsed_s': perf_counter()-start,
        'inventory_sha256': parent['inventory_sha256'], 'balanced_assignment_sha256': control['assignment_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT / (k+'_rule.json')) for k in rules},
        'goal_achieved': False, 'scope': 'Same frozen CuRe;fixed stronger fit+calibration support;exposed development'})


if __name__ == '__main__':
    main()
