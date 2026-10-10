"""Fixed conventional heads on the complete frozen B-Free vector campaign."""

import json
from time import perf_counter

from palimpsest.detection.representations.frozen_bfree import FEATURE_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.bfree_readout.prepare_features import OUTPUT, code_pins
from experiments.origin_detection.semantic_gaussian.run_iteration import null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold

CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def main():
    if (OUTPUT/'iteration.json').exists():
        raise FileExistsError('Preserve B-Free readout campaign')
    receipt = json.loads((OUTPUT/'features.json').read_text())
    if (receipt['code_pins'] != code_pins() or receipt['records'] != 15120
            or receipt['csv_sha256'] != file_sha256(OUTPUT/'features.csv')):
        raise ValueError('B-Free full feature evidence changed')
    old, inventory, parent, _ = parent_data()
    del old
    if receipt['inventory_sha256'] != parent['inventory_sha256']:
        raise ValueError('B-Free source roles changed')
    rows = read_rows(OUTPUT/'features.csv')
    for role in ('fit', 'threshold', 'selection'):
        validate_feature_cache([r for r in rows if r['role'] == role], [r for r in inventory if r['role'] == role],
            FEATURE_NAMES, variants=VARIANTS if role == 'selection' else VARIANTS[:2],
            bounds=(-float('inf'), float('inf')))
    assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Balanced B-Free assignment changed')
    start = perf_counter()
    results, rules, scores = fit_readouts(rows, [('source', FEATURE_NAMES, .1, False),
        ('mean', FEATURE_NAMES, 0, False)], fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'], config=CONFIG, calibrate=class_threshold, score=score_rule)
    sham = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
            if r['role'] == 'fit' else r for r in rows]
    r, fitted, margins = fit_readouts(sham, [('source_null', FEATURE_NAMES, .1, False)],
        fit_variants=VARIANTS[:2], manifest_sha=parent['inventory_sha256'], config=CONFIG,
        calibrate=class_threshold, score=score_rule)
    results.update(r)
    rules.update(fitted)
    scores.update(margins)
    for key, rule in rules.items():
        rule.save(OUTPUT/(key+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json', scores)
    write_json(OUTPUT/'iteration.json', {'candidates': results, 'code_pins': code_pins(),
        'features_receipt_sha256': file_sha256(OUTPUT/'features.json'), 'elapsed_s': perf_counter()-start,
        'inventory_sha256': parent['inventory_sha256'], 'balanced_assignment_sha256': control['assignment_sha256'],
        'null_raw_processed_auc_ci95': null_intervals(scores['source_null']),
        'selection_scores_sha256': file_sha256(OUTPUT/'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT/(k+'_rule.json')) for k in rules},
        'goal_achieved': False, 'scope': 'Released frozen B-Free representation plus conventional head;exposed development'})


if __name__ == '__main__':
    main()
