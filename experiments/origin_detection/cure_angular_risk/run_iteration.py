"""Fixed frozen CuRe direction and radius conventional readouts."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.representations.angular_features import AngularFeatures
from palimpsest.detection.representations.frozen_cure import FULL_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs, null_intervals
from experiments.origin_detection.semantic_covariance_risk.run_iteration import META
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold

OUTPUT = WORK_DIR / 'robust_statistics/cure_angular_risk'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    paths += [REPO_ROOT / p for p in (
        'src/palimpsest/detection/representations/angular_features.py',
        'src/palimpsest/evaluation/source_readout_campaign.py',
        'src/palimpsest/evaluation/source_training.py', 'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/detection/algorithms/source_view_risk.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_angular_features.py', 'tests/detection/test_source_view_risk.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def controls_gate():
    path = OUTPUT/'software_controls.json'
    controls = json.loads(path.read_text())
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Angular planted controls changed or failed')
    return path


def map_rows(rows, mapper):
    values = np.array([[float(r[n]) for n in FULL_NAMES] for r in rows])
    transformed = mapper.transform(values)
    # Exact operation order must survive the single-query deployment path.
    if not np.array_equal(transformed, np.vstack([mapper.transform([r]) for r in values])):
        raise ValueError('Single/batch angular mapping differs')
    return [{**{k: r[k] for k in META}, **dict(zip(mapper.output_names, v.tolist()))}
            for r, v in zip(rows, transformed)]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT/'iteration.json').exists():
        raise FileExistsError('Preserve angular campaign')
    controls = controls_gate()
    rows, parent = cure_inputs()
    assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Balanced angular assignment changed')
    start = perf_counter()
    results, rules, scores, intervals, maps = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for key, include_radius in [('direction', False), ('direction_radius', True)]:
            mapper = AngularFeatures(FULL_NAMES, include_radius)
            mapped = map_rows(rows, mapper)
            r, fitted, margins = fit_readouts(mapped, [(key+'/source', mapper.output_names, .1, False),
                (key+'/mean', mapper.output_names, 0, False)], fit_variants=VARIANTS[:2],
                manifest_sha=parent['inventory_sha256'], config=CONFIG, calibrate=class_threshold, score=score_rule)
            results.update(r)
            rules.update(fitted)
            scores.update(margins)
            sham = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
                    if r['role'] == 'fit' else r for r in mapped]
            null_key = key+'/source_null'
            r, fitted, margins = fit_readouts(sham, [(null_key, mapper.output_names, .1, False)],
                fit_variants=VARIANTS[:2], manifest_sha=parent['inventory_sha256'], config=CONFIG,
                calibrate=class_threshold, score=score_rule)
            results.update(r)
            rules.update(fitted)
            scores.update(margins)
            intervals[null_key] = null_intervals(margins[null_key])
            maps[key] = mapper
    for key, rule in rules.items():
        rule.save(OUTPUT/(key.replace('/', '_')+'_rule.json'))
    for key, mapper in maps.items():
        mapper.save(OUTPUT/(key+'_map.json'))
    write_json(OUTPUT/'selection_scores.json', scores)
    write_json(OUTPUT/'iteration.json', {'candidates': results, 'code_pins': code_pins(),
        'software_controls_sha256': file_sha256(controls), 'elapsed_s': perf_counter()-start,
        'inventory_sha256': parent['inventory_sha256'], 'balanced_assignment_sha256': control['assignment_sha256'],
        'null_raw_processed_auc_ci95': intervals,
        'selection_scores_sha256': file_sha256(OUTPUT/'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT/(k.replace('/', '_')+'_rule.json')) for k in rules},
        'map_files': {k: file_sha256(OUTPUT/(k+'_map.json')) for k in maps}, 'goal_achieved': False,
        'scope': 'Frozen CuRe direction/radius;weak support;exposed development;no physical invariance assertion'})


if __name__ == '__main__':
    main()
