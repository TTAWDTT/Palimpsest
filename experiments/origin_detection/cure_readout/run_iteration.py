"""Six registered conventional heads on frozen released CuRe features."""

from collections import defaultdict
import json
from time import perf_counter

import numpy as np

from palimpsest.detection.representations.frozen_cure import FULL_NAMES, PROJECTED_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.cure_readout.prepare_features import OUTPUT, code_pins, audit_cache
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def main():
    destination = OUTPUT / 'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve frozen CuRe readout evaluation')
    old, inventory, parent, _ = parent_data()
    del old
    receipt = json.loads((OUTPUT / 'features.json').read_text())
    if (receipt['code_pins'] != code_pins() or receipt['csv_sha256'] != file_sha256(OUTPUT / 'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256']
            or not receipt['original_probability_exact'] or receipt['matched_original_probabilities'] != 5040
            or not receipt['wrong_probability_rejected'] or receipt['records'] != 15120):
        raise ValueError('Frozen CuRe feature evidence changed')
    rows = read_rows(OUTPUT / 'features.csv')
    audit_cache(rows, inventory)
    start = perf_counter()
    candidates = [(prefix + '/' + key, names, t, False) for prefix, names in
        [('full', FULL_NAMES), ('projected', PROJECTED_NAMES)] for key, t in [('source', .1), ('mean', 0)]]
    results, rules, scores = fit_readouts(rows, candidates, fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'], config=CONFIG, calibrate=class_threshold, score=score_rule)
    assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Frozen CuRe sham assignment differs')
    sham = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
            if r['role'] == 'fit' else r for r in rows]
    nulls = [('full/source_null', FULL_NAMES, .1, False), ('projected/source_null', PROJECTED_NAMES, .1, False)]
    extra_results, extra_rules, extra_scores = fit_readouts(sham, nulls, fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'], config=CONFIG, calibrate=class_threshold, score=score_rule)
    results.update(extra_results)
    rules.update(extra_rules)
    scores.update(extra_scores)
    intervals = {}
    for null, _, _, _ in nulls:
        groups = defaultdict(list)
        for row in scores[null]:
            if row['variant'] == 'raw' and row['condition'] != 'original':
                groups[row['domain'] + '/' + row['condition']].append(row)
        intervals[null] = {}
        for key, selected in groups.items():
            selected.sort(key=lambda r: r['src'])
            intervals[null][key] = auc_intervals({key: np.array([r['score'] for r in selected])},
                np.array([int(r['label'] == 'FAKE') for r in selected]), CONFIG)[key]
    for key, rule in rules.items():
        rule.save(OUTPUT / (key.replace('/', '_') + '_rule.json'))
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(destination, {'candidates': results, 'null_raw_processed_auc_ci95': intervals,
        'elapsed_s': perf_counter()-start, 'code_pins': code_pins(),
        'inventory_sha256': parent['inventory_sha256'], 'features_sha256': receipt['csv_sha256'],
        'balanced_assignment_sha256': control['assignment_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT / (k.replace('/', '_') + '_rule.json')) for k in rules},
        'goal_achieved': False, 'scope': 'Frozen CuRe representation with locally fitted conventional heads;exposed development'})


if __name__ == '__main__':
    main()
