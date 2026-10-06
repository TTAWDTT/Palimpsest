"""Four fixed source-support controls under a complete paired covariance metric."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.paired_metric import (
    PairedMetricSupportRule, fit_query_map, transform_rows,
)
from palimpsest.detection.algorithms.source_support import fit_source_support
from palimpsest.detection.representations.frozen_cure import FULL_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs, null_intervals
from experiments.origin_detection.paired_kernel.run_iteration import wrong_sources
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT = WORK_DIR / 'robust_statistics/paired_metric_support'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    files = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md']
    files += [REPO_ROOT / p for p in (
        'src/palimpsest/detection/algorithms/paired_metric.py',
        'src/palimpsest/detection/algorithms/source_support.py',
        'src/palimpsest/evaluation/source_training.py', 'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/features.py', 'src/palimpsest/evaluation/robust_views.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/paired_kernel/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_paired_metric.py', 'tests/detection/test_source_support.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def main():
    if (OUTPUT / 'iteration.json').exists():
        raise FileExistsError('Preserve paired metric campaign')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text(encoding='utf-8'))
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Paired metric artificial controls changed')
    rows, parent = cure_inputs()
    fit_records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in VARIANTS[:2]],
                         key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    assignment, null = balanced_source_null(fit_records, seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Balanced metric assignment changed')
    x, truth, weights, sources = weighted_source_arrays(rows, FULL_NAMES, VARIANTS[:2],
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    sham = np.array([assignment[r['domain'], r['src']] for r in fit_records])
    if len(x) != 7560 or len(rows) != 15120:
        raise ValueError('Fixed CuRe weak support coverage changed')
    wrong = wrong_sources(rows, sources)
    views = {key + '/' + variant: view for variant in VARIANTS[:2] for processed in (False, True)
             for key, view in feature_views(rows, FULL_NAMES, 'threshold', processed=processed, variant=variant).items()}
    if len(views) != 24:
        raise ValueError('Fixed metric threshold views changed')
    results, scores, artifacts, maps = {}, {}, {}, {}
    start = perf_counter()
    with threadpool_limits(limits=1):
        for mode, paired, grouping in (('euclidean', False, sources), ('paired', True, sources),
                                       ('wrong_source', True, wrong)):
            maps[mode] = fit_query_map(x, weights, grouping, paired=paired)
        print(json.dumps({'maps_ready_s': perf_counter() - start}), flush=True)
        for key, mode, labels in (('euclidean', 'euclidean', truth), ('paired', 'paired', truth),
                                  ('source_null', 'paired', sham), ('wrong_source', 'wrong_source', truth)):
            center, scale, transform, diagnostics = maps[mode]
            support = fit_source_support(transform_rows(x, center, scale, transform), labels, sources,
                feature_names=FULL_NAMES, metric_weights=(1.,) * len(FULL_NAMES), neighbors=5,
                manifest_sha=parent['inventory_sha256'])
            rule = PairedMetricSupportRule(FULL_NAMES, center, scale, transform, support)
            rule, calibration = class_threshold(rule, views)
            path = OUTPUT / (key + '_rule.json')
            rule.save(path)
            restored = PairedMetricSupportRule.load(path)
            probe = x[:12]
            if not np.array_equal(rule.score(probe), restored.score(probe)):
                raise ValueError('Paired metric loaded rule changed')
            margins, result = score_rule(rows, FULL_NAMES, rule, CONFIG)
            result.update({'threshold': rule.threshold, 'calibration': calibration, 'metric_diagnostics': diagnostics})
            results[key], scores[key] = result, margins
            artifacts[key] = {p.name: file_sha256(p) for p in sorted(OUTPUT.glob(key + '_rule*'))}
            print(json.dumps({'candidate': key, 'minimum_domain_ba': result['minimum_domain_ba'],
                'minimum_scene_ba': result['minimum_scene_ba'],
                'maximum_drop': result['maximum_any_scene_drop']}), flush=True)
            del rule, restored, support
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(OUTPUT / 'iteration.json', {'candidates': results, 'code_pins': code_pins(),
        'inventory_sha256': parent['inventory_sha256'], 'balanced_assignment_sha256': null['assignment_sha256'],
        'elapsed_s': perf_counter() - start, 'null_raw_processed_auc_ci95': null_intervals(scores['source_null']),
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'), 'rule_files': artifacts,
        'goal_achieved': False, 'scope': 'Full paired distance on frozen CuRe;exposed development;not XQDA reproduction'})


if __name__ == '__main__':
    main()
