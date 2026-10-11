"""Frozen RGB dependence diagnostic, never decoding RR reserved sources."""

import argparse
from fractions import Fraction
import json
import random
from time import perf_counter
import tomllib

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.color_relations import FEATURE_NAMES, extract_colors
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.pixel_features import audit_variants, extract_inventory, join_features
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.conditional_residual.run_iteration import final_target_screen
from experiments.origin_detection.gaussian_readout.run_iteration import signed_inputs, NAMES
from experiments.origin_detection.kernel_readout.run_iteration import fit_rule
from experiments.origin_detection.paired_stability.run_iteration import INTERVENTION
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.residual_statistics.run_iteration import screen_rule, aggregate
from experiments.origin_detection.residual_training.fit_rules import TRAINING_VARIANT
from experiments.origin_detection.threshold_diagnostics.audit_rate_boundaries import audit

CONFIG = REPO_ROOT/'configs/evaluation/color_relations.toml'
OUTPUT = WORK_DIR/'robust_statistics/color_relations'
ORDINARY = ('raw', TRAINING_VARIANT)
MODES = {'color': FEATURE_NAMES, 'hybrid_color': NAMES['hybrid']+FEATURE_NAMES}


def code_pins():
    files = [CONFIG, REPO_ROOT/'src/palimpsest/detection/algorithms/color_relations.py',
             REPO_ROOT/'src/palimpsest/evaluation/pixel_features.py',
             REPO_ROOT/'tests/detection/test_color_relations.py',
             REPO_ROOT/'experiments/origin_detection/kernel_readout/run_iteration.py',
             REPO_ROOT/'experiments/origin_detection/gaussian_readout/run_iteration.py']
    directory = REPO_ROOT/'experiments/origin_detection/color_relations'
    files.extend(directory.glob('*.py')); files.append(directory/'README.md')
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', type=int, default=0)
    group.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.pilot < 0: parser.error('Pilot must be positive')
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    control_path = OUTPUT/'controls.json'
    control = json.loads(control_path.read_text(encoding='utf-8'))
    if control['returncode'] != 0 or control['test_sha256'] != file_sha256(REPO_ROOT/'tests/detection/test_color_relations.py'):
        raise ValueError('Color controls failed or changed')
    ordinary, held, parent = signed_inputs()
    removed = set(NAMES['hybrid']) | {'preprocess_ms', 'statistics_ms', 'variant'}
    inventory = [{k: v for k, v in r.items() if k not in removed} for r in ordinary if r['variant'] == 'raw']
    cv2.setNumThreads(1)
    if args.prepare or args.pilot:
        directory = OUTPUT if args.prepare else OUTPUT.with_name('color_relations_pilot')
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/'features.json').exists(): raise FileExistsError('Preserve color extraction receipt')
        if args.pilot:
            random.Random(config['seed']).shuffle(inventory); inventory = inventory[:args.pilot]
        result = extract_inventory(inventory, directory/'features.csv', names=FEATURE_NAMES, extractor=extract_colors,
                                   resolve_path=image_path, resize=resize256, ordinary_variants=ORDINARY,
                                   selection_variant=INTERVENTION, jpeg_parameters={TRAINING_VARIANT: (90, 0), INTERVENTION: (70, 2)})
        write_json(directory/'features.json', {**result, 'code_pins': code_pins(),
                    'inventory_sha256': parent['inventory_sha256'], 'controls_sha256': file_sha256(control_path),
                    'parent_features_receipt_sha256': file_sha256(WORK_DIR/'robust_statistics/wavelet_envelopes/features.json')})
        return
    if (OUTPUT/'iteration.json').exists(): raise FileExistsError('Preserve color evaluation')
    receipt = json.loads((OUTPUT/'features.json').read_text(encoding='utf-8'))
    if (receipt['code_pins'] != code_pins() or receipt['csv_sha256'] != file_sha256(OUTPUT/'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256']):
        raise ValueError('Color code/cache/inventory changed')
    raw, extra = audit_variants(read_rows(OUTPUT/'features.csv'), inventory, FEATURE_NAMES,
                                ordinary_variants=ORDINARY, selection_variant=INTERVENTION)
    rows, unseen = join_features(ordinary, raw, FEATURE_NAMES), join_features(held, extra, FEATURE_NAMES)
    start = perf_counter(); results = {}; rules = {}
    candidates = [(rep, objective, False) for rep in config['candidate_representations'] for objective in config['candidate_objectives']]
    candidates.append(('hybrid_color', 'both', True))
    with threadpool_limits(limits=1):
        for rep, objective, shuffled in candidates:
            key = rep+'/'+objective+('_source_shuffled_null' if shuffled else '')
            names = MODES[rep]
            rule, diagnostic = fit_rule(rows, names, objective, config, parent['inventory_sha256'], shuffled=shuffled)
            result = screen_rule(rows, names, rule, config)
            encoded = aggregate(unseen, names, rule, INTERVENTION)
            changes = [(v['balanced_accuracy_at_zero'], encoded[d]['conditions'][c]['balanced_accuracy_at_zero'])
                       for d, z in result['selection_aggregate'].items() for c, v in z['conditions'].items() if c != 'original']
            result['criteria'].update({
                'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
                'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
                'both_encoded_class_accuracies': all(v[k] >= .55 for data in (result['recompression_aggregate'], encoded)
                                                     for z in data.values() for v in z['conditions'].values()
                                                     for k in ('real_accuracy_at_zero', 'fake_accuracy_at_zero'))})
            result.update({'held_encoding_aggregate': encoded, 'fit_diagnostics': diagnostic,
                           'final_target': final_target_screen(result, encoded, config)})
            exact = audit({'candidates': {key: result}}, Fraction(str(config['maximum_reencoded_ba_drop'])))[key]
            result['floating_point_criteria'] = result['criteria']; result['criteria'] = exact['corrected_criteria']
            result['gate_passed'] = exact['corrected_gate_passed']; result['rate_boundary_audit'] = exact['comparisons']
            for variant, data in [('raw', rows), (TRAINING_VARIANT, rows), (INTERVENTION, unseen)]:
                matrix = np.array([[float(r[n]) for n in names] for r in data if r['role'] == 'selection' and r['variant'] == variant])
                if not np.array_equal(rule.score(matrix), [rule.score([row])[0] for row in matrix]):
                    raise ValueError('Color single/batch score differs')
            results[key] = result; rules[key] = rule
            print(json.dumps({'candidate': key, 'gate': result['gate_passed'], 'target': result['final_target']}), flush=True)
    null = 'hybrid_color/both_source_shuffled_null'
    refused = results[null]['criteria']['eight_auc_lower_bounds']
    passing = [k for k, v in results.items() if k != null and v['gate_passed']]
    chosen = max(passing, key=lambda k: results[k]['worst_processed_auc']) if passing and not refused else None
    for key, rule in rules.items(): rule.save(OUTPUT/(key.replace('/', '_')+'_rule.json'))
    write_json(OUTPUT/'iteration.json', {'candidates': results, 'development_chosen': chosen,
                'interpretation_refused': refused, 'elapsed_s': perf_counter()-start,
                'single_batch_scores_exact': True, 'code_pins': code_pins(),
                'inventory_sha256': parent['inventory_sha256'], 'controls_sha256': file_sha256(control_path),
                'features_receipt_sha256': file_sha256(OUTPUT/'features.json'),
                'rule_files': {k: file_sha256(OUTPUT/(k.replace('/', '_')+'_rule.json')) for k in rules},
                'scope': 'Fixed RGB representation diagnostic on exposed development, not independent validation'})


if __name__ == '__main__':
    main()
