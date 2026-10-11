"""Conventional fixed heads on audited canonicalization features."""

from collections import defaultdict
import json
from time import perf_counter

import numpy as np

from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.pixel_features import audit_variants
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.canonical_jpeg.prepare_features import OUTPUT, code_pins
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS, Q60
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def main():
    destination = OUTPUT / 'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve canonical head evaluation')
    old, inventory, parent, _ = parent_data()
    del old
    receipt = json.loads((OUTPUT / 'features.json').read_text())
    if (receipt['code_pins'] != code_pins() or receipt['csv_sha256'] != file_sha256(OUTPUT / 'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256']
            or not receipt['raw_to_q70']['exact_equal'] or not receipt['wrong_value_rejected']):
        raise ValueError('Canonical cache changed')
    rows = read_rows(OUTPUT / 'features.csv')
    audit_variants(rows, inventory, FEATURE_NAMES, ordinary_variants=VARIANTS[:3], selection_variant=Q60)
    start = perf_counter()
    results, rules, scores = fit_readouts(rows,
        [('canonical/source', FEATURE_NAMES, .1, False), ('canonical/mean', FEATURE_NAMES, 0, False)],
        fit_variants=VARIANTS[:2], manifest_sha=parent['inventory_sha256'],
        config=CONFIG, calibrate=class_threshold, score=score_rule)
    assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Canonical sham assignment differs')
    sham = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
            if r['role'] == 'fit' else r for r in rows]
    extra_results, extra_rules, extra_scores = fit_readouts(sham,
        [('canonical/source_null', FEATURE_NAMES, .1, False)], fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'], config=CONFIG, calibrate=class_threshold, score=score_rule)
    results.update(extra_results)
    rules.update(extra_rules)
    scores.update(extra_scores)
    groups = defaultdict(list)
    for row in scores['canonical/source_null']:
        if row['variant'] == 'raw' and row['condition'] != 'original':
            groups[row['domain'] + '/' + row['condition']].append(row)
    intervals = {}
    for key, records in groups.items():
        records.sort(key=lambda r: r['src'])
        intervals[key] = auc_intervals({key: np.array([r['score'] for r in records])},
            np.array([int(r['label'] == 'FAKE') for r in records]), CONFIG)[key]
    for key, rule in rules.items():
        rule.save(OUTPUT / (key.replace('/', '_') + '_rule.json'))
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(destination, {'candidates': results, 'null_raw_processed_auc_ci95': intervals,
        'elapsed_s': perf_counter() - start, 'code_pins': code_pins(),
        'inventory_sha256': parent['inventory_sha256'], 'features_sha256': receipt['csv_sha256'],
        'balanced_assignment_sha256': control['assignment_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT / (k.replace('/', '_') + '_rule.json')) for k in rules},
        'goal_achieved': False, 'scope': 'Fixed common JPEG plus frozen CLIP and conventional heads;exposed development'})


if __name__ == '__main__':
    main()
