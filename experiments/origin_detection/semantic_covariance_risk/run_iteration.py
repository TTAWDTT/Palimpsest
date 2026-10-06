"""Quadratic source-centroid variance modes under a same-source loss."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.covariance_modes import fit_covariance_modes
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES as CLIP_NAMES
from palimpsest.detection.representations.frozen_cure import FULL_NAMES as CURE_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_kernel.run_iteration import inputs as clip_inputs
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs, null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold

OUTPUT = WORK_DIR / 'robust_statistics/semantic_covariance_risk'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}
META = ('domain', 'scene', 'condition', 'variant', 'src', 'role', 'label')


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md']
    paths += [REPO_ROOT / p for p in (
        'src/palimpsest/detection/algorithms/covariance_modes.py',
        'src/palimpsest/evaluation/source_training.py', 'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/source_readout_campaign.py',
        'src/palimpsest/detection/algorithms/source_view_risk.py',
        'src/palimpsest/evaluation/robust_views.py', 'src/palimpsest/evaluation/features.py',
        'experiments/origin_detection/semantic_kernel/run_iteration.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_covariance_modes.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def mapped_rows(rows, names, manifest_sha):
    x, y, weights, sources = weighted_source_arrays(rows, names, VARIANTS[:2],
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    mapper, diagnostic = fit_covariance_modes(x, y, weights, sources, feature_names=names,
        count=32, ridge=.1, shrinkage=.5, scale_floor=.001, manifest_sha=manifest_sha)
    values = np.array([[float(r[n]) for n in names] for r in rows])
    transformed = mapper.transform(values)
    selected = np.array([i for i, r in enumerate(rows) if r['role'] == 'selection'])
    if not np.array_equal(transformed[selected], mapper.transform(values[selected])):
        raise ValueError('Selection mapped features changed on repeated traversal')
    mapped = [{**{k: r[k] for k in META}, **dict(zip(mapper.output_names, v.tolist()))}
              for r, v in zip(rows, transformed)]
    return mapped, mapper, diagnostic


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'iteration.json').exists():
        raise FileExistsError('Preserve quadratic source-risk campaign')
    controls_path = OUTPUT / 'software_controls.json'
    controls = json.loads(controls_path.read_text())
    if not controls['passed'] or any(file_sha256(REPO_ROOT / k) != v for k, v in controls['test_pins'].items()):
        raise ValueError('Planted covariance controls changed')
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    results, rules, maps, scores, intervals, map_diagnostics = {}, {}, {}, {}, {}, {}
    manifest_sha = None
    start = perf_counter()
    with threadpool_limits(limits=1):
        for prefix, names, loader in [('clip', CLIP_NAMES, clip_inputs), ('cure', CURE_NAMES, cure_inputs)]:
            rows, parent = loader()
            if manifest_sha is not None and manifest_sha != parent['inventory_sha256']:
                raise ValueError('Frozen source inventory differs')
            manifest_sha = parent['inventory_sha256']
            assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
            if control['assignment_sha256'] != prior['assignment_sha256']:
                raise ValueError('Balanced source assignment changed')
            mapped, mapper, diagnostic = mapped_rows(rows, names, manifest_sha)
            maps[prefix], map_diagnostics[prefix] = mapper, diagnostic
            candidates = [(prefix+'/source', mapper.output_names, .1, False),
                          (prefix+'/mean', mapper.output_names, 0, False),
                          (prefix+'/quadratic_source', mapper.quadratic_names, .1, False)]
            r, fitted, margins = fit_readouts(mapped, candidates, fit_variants=VARIANTS[:2],
                manifest_sha=manifest_sha, config=CONFIG, calibrate=class_threshold, score=score_rule)
            results.update(r)
            rules.update(fitted)
            scores.update(margins)
            del mapped
            sham = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
                    if r['role'] == 'fit' else r for r in rows]
            mapped, mapper, diagnostic = mapped_rows(sham, names, manifest_sha)
            null_key = prefix+'/source_null'
            maps[prefix+'_null'], map_diagnostics[prefix+'_null'] = mapper, diagnostic
            r, fitted, margins = fit_readouts(mapped, [(null_key, mapper.output_names, .1, False)],
                fit_variants=VARIANTS[:2], manifest_sha=manifest_sha, config=CONFIG,
                calibrate=class_threshold, score=score_rule)
            results.update(r)
            rules.update(fitted)
            scores.update(margins)
            intervals[null_key] = null_intervals(margins[null_key])
            del rows, sham, mapped
    for key, rule in rules.items():
        rule.save(OUTPUT / (key.replace('/', '_')+'_rule.json'))
    for key, mapper in maps.items():
        mapper.save(OUTPUT / (key+'_map.json'))
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(OUTPUT / 'iteration.json', {'candidates': results, 'code_pins': code_pins(),
        'software_controls_sha256': file_sha256(controls_path), 'map_diagnostics': map_diagnostics,
        'map_files': {k: file_sha256(OUTPUT / (k+'_map.json')) for k in maps},
        'null_raw_processed_auc_ci95': intervals, 'elapsed_s': perf_counter()-start,
        'inventory_sha256': manifest_sha, 'balanced_assignment_sha256': control['assignment_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT / (k.replace('/', '_')+'_rule.json')) for k in rules},
        'goal_achieved': False, 'scope': 'Fit-only centroid covariance modes and conventional source-risk head;exposed development'})


if __name__ == '__main__':
    main()
