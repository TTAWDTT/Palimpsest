"""Prospective nuisance alignment; signed development caches, no new pixels."""

import argparse
import json
from time import perf_counter
import tomllib

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.conditional_residual import gate_values
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.paired_stability.run_iteration import load_inputs, INTERVENTION
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.residual_statistics.run_iteration import screen_rule, aggregate
from experiments.origin_detection.residual_training.fit_rules import TRAINING_VARIANT
from .fit_rules import prepare_gate, fit_rule

CONFIG = REPO_ROOT/'configs/evaluation/conditional_residual.toml'
OUTPUT = WORK_DIR/'robust_statistics/conditional_residual'


def code_pins():
    directory = REPO_ROOT/'experiments/origin_detection/conditional_residual'
    files = [CONFIG, REPO_ROOT/'src/palimpsest/detection/algorithms/conditional_residual.py',
             REPO_ROOT/'tests/detection/test_conditional_residual.py', directory/'README.md', *directory.glob('*.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def gate_diagnostics(rows, held, gate):
    groups = {}
    for r in rows:
        if r['role'] == 'selection':
            key = '/'.join(r[k] for k in ('domain', 'scene', 'condition', 'label'))
            groups.setdefault(key, []).append(r)
    result = {}
    for key, records in sorted(groups.items()):
        x = [[float(r[n]) for n in FEATURE_NAMES] for r in records]
        scores = gate.score(x)
        labels = np.array([r['variant'] == TRAINING_VARIANT for r in records], int)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError('Incomplete nuisance gate diagnostic')
        result[key] = {'records': len(records), 'auc': roc_auc_score(labels, scores),
                       'balanced_accuracy': float(((scores > 0) == labels).mean())}
    unseen = {}
    for r in held:
        key = '/'.join(r[k] for k in ('domain', 'condition', 'label'))
        unseen.setdefault(key, []).append(r)
    return {'known90_within_truth_class': result,
            'unseen70_gate_mean': {k: float(gate_values(gate, [[float(r[n]) for n in FEATURE_NAMES]
                                                               for r in z]).mean()) for k, z in sorted(unseen.items())},
            'scope': 'Pixel-only extra-JPEG classifier on exposed development; not physical history recovery'}


def final_target_screen(result, encoded, config):
    """Provisional numerical target only; independent validation remains OPEN."""
    raw, seen = result['selection_aggregate'], result['recompression_aggregate']
    ba = lambda v: v['balanced_accuracy_at_zero']
    accuracies = [ba(v) for data in (raw, seen, encoded) for domain in data.values()
                  for v in domain['conditions'].values()]
    drops = []
    for d in raw:
        base = ba(raw[d]['conditions']['original'])
        for c, v in raw[d]['conditions'].items():
            if c != 'original': drops.append(base-ba(v))
            for data in (seen, encoded):
                drops.append(ba(v)-ba(data[d]['conditions'][c]))
    return {'absolute_ba': min(accuracies) >= config['provisional_final_ba'],
            'paired_drop': max(drops) <= config['provisional_final_drop'],
            'minimum_ba': min(accuracies), 'maximum_observed_drop': max(drops),
            'independent_real_validation': False, 'speed_validated': False,
            'verdict': 'OPEN: development evidence cannot close the research goal'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    path = OUTPUT/('pilot.json' if args.pilot else 'iteration.json')
    if path.exists(): raise FileExistsError('Preserve conditional residual result')
    controls = OUTPUT/'controls.json'
    if not controls.exists(): raise ValueError('Run and record planted controls before fitting')
    control = json.loads(controls.read_text(encoding='utf-8'))
    if control.get('returncode') != 0 or control.get('test_sha256') != file_sha256(
            REPO_ROOT/'tests/detection/test_conditional_residual.py'):
        raise ValueError('Conditional controls failed or changed')
    rows, held, receipt = load_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with threadpool_limits(limits=1):
        prepared = prepare_gate(rows, config, receipt['inventory_sha256'])
        if args.pilot:
            _, fit = fit_rule(rows, prepared, 0., config, receipt['inventory_sha256'])
            result = {'pilot_fit': fit, 'full_width_fits': 1, 'planned_origin_fits': 3,
                      'scope': 'Full gate preparation and one origin fit; evaluation cost excluded'}
        else:
            result = {'gate_diagnostics': gate_diagnostics(rows, held, prepared[0]), 'candidates': {}}
            rules = {}
            for strength, shuffled in [(0., False), (1., False), (0., True)]:
                name = 'source_shuffled_null' if shuffled else f'lambda{strength:g}'
                rule, fit = fit_rule(rows, prepared, strength, config, receipt['inventory_sha256'], shuffled=shuffled)
                value = screen_rule(rows, FEATURE_NAMES, rule, config)
                encoded = aggregate(held, FEATURE_NAMES, rule, INTERVENTION)
                changes = [(v['balanced_accuracy_at_zero'], encoded[d]['conditions'][c]['balanced_accuracy_at_zero'])
                           for d, z in value['selection_aggregate'].items()
                           for c, v in z['conditions'].items() if c != 'original']
                value['criteria'].update({
                    'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
                    'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
                    'both_encoded_class_accuracies': all(v[k] >= .55 for data in (value['recompression_aggregate'], encoded)
                                                         for z in data.values() for v in z['conditions'].values()
                                                         for k in ('real_accuracy_at_zero', 'fake_accuracy_at_zero'))})
                value['gate_passed'] = all(value['criteria'].values())
                value.update({'held_encoding_aggregate': encoded, 'fit_diagnostics': fit,
                              'final_target': final_target_screen(value, encoded, config)})
                for variant, data in [('raw', rows), (TRAINING_VARIANT, rows), (INTERVENTION, held)]:
                    selected = [r for r in data if r['role'] == 'selection' and r['variant'] == variant]
                    x = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in selected])
                    if not np.array_equal(rule.score(x), [rule.score([v])[0] for v in x]):
                        raise ValueError('Conditional single/batch score differs')
                result['candidates'][name] = value
                rules[name] = rule
                print(json.dumps({'candidate': name, 'development_gate': value['gate_passed'],
                                  'target': value['final_target']}), flush=True)
            passing = [k for k, v in result['candidates'].items() if k != 'source_shuffled_null' and v['gate_passed']]
            refused = result['candidates']['source_shuffled_null']['criteria']['eight_auc_lower_bounds']
            result.update({'development_chosen': max(passing, key=lambda k: result['candidates'][k]['worst_processed_auc'])
                                               if passing and not refused else None,
                           'interpretation_refused': refused, 'single_batch_scores_exact': True,
                           'scope': 'Repeated RR/Chimera development; no independent final verification'})
            for k, rule in rules.items(): rule.save(OUTPUT/f'{k}_rule.json')
            result['rule_files'] = {k: file_sha256(OUTPUT/f'{k}_rule.json') for k in rules}
    result.update({'elapsed_s': perf_counter()-start, 'code_pins': code_pins(),
                   'controls_sha256': file_sha256(controls),
                   'parent_features_receipt_sha256': file_sha256(WORK_DIR/'robust_statistics/joint_residual_statistics/features.json'),
                   'parent_intervention_receipt_sha256': file_sha256(WORK_DIR/'robust_statistics/residual_training/features.json'),
                   'inventory_sha256': receipt['inventory_sha256']})
    write_json(path, result)


if __name__ == '__main__': main()
