"""Fixed local Gaussian scores and true-source variance penalties on signed CLIP."""

from collections import defaultdict
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.paired_kernel import (
    PairedKernelRule, gaussian_rows, solve_panel, source_centers,
)
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_kernel.run_iteration import inputs
from experiments.origin_detection.semantic_gaussian.run_iteration import null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT = WORK_DIR/'robust_statistics/paired_kernel'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    files += [REPO_ROOT/p for p in (
        'src/palimpsest/detection/algorithms/paired_kernel.py',
        'src/palimpsest/evaluation/source_training.py',
        'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/robust_views.py',
        'experiments/origin_detection/semantic_kernel/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_paired_kernel.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def wrong_sources(rows, sources):
    records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in VARIANTS[:2]],
                     key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    if len(records) != len(sources):
        raise ValueError('Wrong-source alignment changed')
    strata = defaultdict(list)
    for i, row in enumerate(records):
        strata[tuple(row[k] for k in ('domain', 'scene', 'label', 'condition', 'variant'))].append(i)
    rng, wrong = random.Random(CONFIG['seed']), sources.copy()
    for key in sorted(strata):
        indices = strata[key]
        values = sources[indices].tolist()
        rng.shuffle(values)
        wrong[indices] = values
    if np.array_equal(wrong, sources) or not np.array_equal(np.bincount(wrong), np.bincount(sources)):
        raise ValueError('Wrong-source control did not alter complete panels')
    return wrong


def main():
    if (OUTPUT/'iteration.json').exists():
        raise FileExistsError('Preserve paired kernel campaign')
    control = json.loads((OUTPUT/'software_controls.json').read_text())
    if not control['passed'] or control['code_pins'] != code_pins():
        raise ValueError('Paired kernel artificial gates changed')
    rows, parent = inputs()
    start = perf_counter()
    assignment, null = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Balanced kernel assignment differs')
    results, scored_all, files = {}, {}, {}
    with threadpool_limits(limits=1):
        x, y, weights, sources = weighted_source_arrays(rows, FEATURE_NAMES, VARIANTS[:2],
            expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
        if len(x) != 7560:
            raise ValueError('Weak kernel panel differs')
        centers, width = source_centers(x, sources)
        phi = gaussian_rows(x, centers, width)
        kaa = gaussian_rows(centers, centers, width)
        wrong = wrong_sources(rows, sources)
        records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in VARIANTS[:2]],
                         key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
        sham = np.array([assignment[r['domain'], r['src']] for r in records])
        print(json.dumps({'anchors': len(centers), 'width': width, 'mapping_s': perf_counter()-start}), flush=True)
        views = {key+'/'+v: view for v in VARIANTS[:2] for processed in (False, True)
                 for key, view in feature_views(rows, FEATURE_NAMES, 'threshold', processed=processed, variant=v).items()}
        if len(views) != 24:
            raise ValueError('Kernel threshold views differ')
        for key, strength, labels, groups in (
            ('zero', 0., y, sources), ('paired', 1., y, sources),
            ('source_null', 1., sham, sources), ('wrong_source', 1., y, wrong),
        ):
            alpha, bias, diagnostic = solve_panel(phi, labels, weights, groups, kaa, strength=strength)
            rule = PairedKernelRule(FEATURE_NAMES, tuple(tuple(a) for a in centers), width,
                                   tuple(alpha), bias, 0., parent['inventory_sha256'])
            rule, diagnostic['calibration'] = class_threshold(rule, views)
            path = OUTPUT/(key+'_rule.json')
            if path.exists():
                raise FileExistsError('Preserve kernel rule artifact')
            rule.save(path)
            restored = PairedKernelRule.load(path)
            np.testing.assert_array_equal(restored.score(x[:12]), rule.score(x[:12]))
            scored, result = score_rule(rows, FEATURE_NAMES, rule, CONFIG)
            result.update({'fit_diagnostics': diagnostic, 'threshold': rule.threshold})
            results[key], scored_all[key], files[key] = result, scored, file_sha256(path)
            print(json.dumps({'candidate': key, 'minimum_ba': result['minimum_domain_ba'],
                'minimum_scene_ba': result['minimum_scene_ba'],
                'maximum_drop': result['maximum_any_scene_drop'], 'fit_diagnostics': diagnostic}), flush=True)
    write_json(OUTPUT/'selection_scores.json', scored_all)
    write_json(OUTPUT/'iteration.json', {'candidates': results, 'elapsed_s': perf_counter()-start,
        'code_pins': code_pins(), 'inventory_sha256': parent['inventory_sha256'], 'width': width,
        'anchors': len(centers), 'balanced_assignment_sha256': null['assignment_sha256'],
        'null_raw_processed_auc_ci95': null_intervals(scored_all['source_null']), 'rule_files': files,
        'selection_scores_sha256': file_sha256(OUTPUT/'selection_scores.json'), 'goal_achieved': False,
        'scope': 'Signed frozen neural CLIP plus restricted traditional paired kernel;exposed development,not independent real validation'})


if __name__ == '__main__':
    main()
