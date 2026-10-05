"""Locate threshold-role class conflicts in frozen paired readouts."""

import json

from palimpsest.detection.algorithms.paired_stability import StableRule
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.thresholds import threshold_interval
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.paired_stability.run_iteration import OUTPUT as PARENT, load_inputs, code_pins


def main():
    output = WORK_DIR/'robust_statistics/threshold_diagnostics.json'
    if output.exists(): raise FileExistsError('Preserve finite threshold diagnostics')
    rows, _, _ = load_inputs()
    parent = json.loads((PARENT/'iteration.json').read_text(encoding='utf-8'))
    if parent['code_pins'] != code_pins(): raise ValueError('Frozen readout changed')
    results = {}
    for mode, digest in parent['rule_files'].items():
        path = PARENT/f'{mode}_rule.json'
        if file_sha256(path) != digest: raise ValueError('Frozen rule changed')
        rule = StableRule.load(path)
        views = {}
        for variant in ('raw', 'jpeg90_444_after_resize256'):
            for processed in (False, True):
                for key, (_, x, y) in feature_views(rows, FEATURE_NAMES, 'threshold', processed=processed, variant=variant).items():
                    views[key+'/'+variant] = threshold_interval(rule.score(x), y)
        lower_key = max(views, key=lambda k: views[k]['lower_inclusive'])
        upper_key = min(views, key=lambda k: views[k]['upper_exclusive'])
        lower, upper = views[lower_key]['lower_inclusive'], views[upper_key]['upper_exclusive']
        results[mode] = {'views': views, 'individually_infeasible': [k for k, v in views.items() if not v['feasible']],
                         'global_lower': lower, 'global_upper': upper, 'global_feasible': lower < upper,
                         'lower_bound_view': lower_key, 'upper_bound_view': upper_key}
    sources = [REPO_ROOT/'src/palimpsest/evaluation/thresholds.py',
               *sorted((REPO_ROOT/'experiments/origin_detection/threshold_diagnostics').glob('*.py'))]
    write_json(output, {'results': results, 'role': 'threshold', 'minimum_class_accuracy': .55,
                        'parent_iteration_sha256': file_sha256(PARENT/'iteration.json'),
                        'code_pins': {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sources}})
    print(json.dumps({m: {k: v for k, v in z.items() if k != 'views'} for m, z in results.items()}, indent=2))


if __name__ == '__main__': main()
