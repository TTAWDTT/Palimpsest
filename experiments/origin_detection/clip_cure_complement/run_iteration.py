"""Fixed signed semantic and adapted representations, without encoder fitting."""

import json
from pathlib import Path
from time import perf_counter

from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES as CLIP_NAMES
from palimpsest.detection.representations.frozen_cure import FULL_NAMES as CURE_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.evaluation.pixel_features import join_features
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_kernel.run_iteration import inputs as clip_inputs
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs, null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold

OUTPUT = WORK_DIR/'robust_statistics/clip_cure_complement'
FEATURE_NAMES = CLIP_NAMES+CURE_NAMES
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    files += [REPO_ROOT/p for p in (
        'src/palimpsest/evaluation/features.py', 'src/palimpsest/evaluation/pixel_features.py',
        'src/palimpsest/evaluation/source_training.py', 'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/source_readout_campaign.py',
        'src/palimpsest/detection/algorithms/source_view_risk.py',
        'src/palimpsest/detection/representations/feature_pair.py',
        'experiments/origin_detection/semantic_kernel/run_iteration.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_feature_pair.py', 'tests/evaluation/test_clip_cure_join.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def audited_join(left, right, inventory, *, first_names, second_names, variants):
    if set(first_names) & set(second_names):
        raise ValueError('Joint feature names overlap')
    for rows, names in ((left, first_names), (right, second_names)):
        validate_feature_cache(rows, inventory, names, variants=variants, bounds=(-float('inf'), float('inf')))
    combined = join_features(left, right, second_names)
    validate_feature_cache(combined, inventory, tuple(first_names)+tuple(second_names),
                           variants=variants, bounds=(-float('inf'), float('inf')))
    return combined


def main():
    if (OUTPUT/'iteration.json').exists():
        raise FileExistsError('Preserve frozen complement campaign')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Joint union artificial controls changed')
    clip, parent = clip_inputs()
    cure, other = cure_inputs()
    _, inventory, signed, _ = parent_data()
    if not (parent['inventory_sha256'] == other['inventory_sha256'] == signed['inventory_sha256']):
        raise ValueError('Joint source lineage differs')
    clip = [r for r in clip if r['role'] == 'selection' or r['variant'] in VARIANTS[:2]]
    joined = []
    for role in ('fit', 'threshold', 'selection'):
        joined += audited_join([r for r in clip if r['role'] == role], [r for r in cure if r['role'] == role],
            [r for r in inventory if r['role'] == role], first_names=CLIP_NAMES, second_names=CURE_NAMES,
            variants=VARIANTS if role == 'selection' else VARIANTS[:2])
    del clip, cure
    if len(joined) != 15120:
        raise ValueError('Joint union record count differs')
    assignment, control = balanced_source_null([r for r in joined if r['role'] == 'fit'], seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Joint balanced assignment changed')
    start = perf_counter()
    results, rules, scores = fit_readouts(joined, [('source', FEATURE_NAMES, .1, False),
        ('mean', FEATURE_NAMES, 0, False)], fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'], config=CONFIG, calibrate=class_threshold, score=score_rule)
    sham = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
            if r['role'] == 'fit' else r for r in joined]
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
        'elapsed_s': perf_counter()-start, 'inventory_sha256': parent['inventory_sha256'],
        'balanced_assignment_sha256': control['assignment_sha256'], 'records': len(joined),
        'descriptor_dimensions': len(FEATURE_NAMES), 'null_raw_processed_auc_ci95': null_intervals(scores['source_null']),
        'selection_scores_sha256': file_sha256(OUTPUT/'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT/(k+'_rule.json')) for k in rules}, 'goal_achieved': False,
        'scope': 'Two frozen neural representations plus traditional heads;exposed development,combined live cost not measured'})


if __name__ == '__main__':
    main()
