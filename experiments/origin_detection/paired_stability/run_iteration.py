"""Signed paired-score consistency experiment; no new image feature extraction."""

import argparse
import json
from time import perf_counter
import tomllib

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.paired_stability import StableDetector
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.residual_statistics.run_iteration import signed_features, screen_rule, aggregate
from experiments.origin_detection.residual_training.run_iteration import (
    OUTPUT as PARENT, INTERVENTION, code_pins as parent_pins, audit_intervention, selection_inventory,
)
from .fit_rules import fit_rule, paired_deltas

CONFIG = REPO_ROOT / 'configs/evaluation/paired_stability.toml'
OUTPUT = WORK_DIR / 'robust_statistics/paired_stability'


def code_pins():
    directory = REPO_ROOT / 'experiments/origin_detection/paired_stability'
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted([
        CONFIG, REPO_ROOT/'src/palimpsest/detection/algorithms/paired_stability.py',
        directory/'README.md', *directory.glob('*.py')])}


def load_inputs():
    rows, receipt = signed_features()
    parent = json.loads((PARENT/'features.json').read_text(encoding='utf-8'))
    if parent['code_pins'] != parent_pins() or parent['csv_sha256'] != file_sha256(PARENT/'features.csv'):
        raise ValueError('Seventh parent intervention changed')
    held = audit_intervention(read_rows(PARENT/'features.csv'), selection_inventory(rows))
    return rows, held, receipt


def screen(rows, held, config, manifest_sha):
    output, rules = {}, {}
    for strength, shuffled in [*( (v, False) for v in config['candidate_strengths']), (config['null_strength'], True)]:
        mode = 'source_shuffled_null' if shuffled else f'lambda{strength:g}'
        rule, fit = fit_rule(rows, strength, config, manifest_sha, shuffled=shuffled)
        result = screen_rule(rows, FEATURE_NAMES, rule, config)
        encoded = aggregate(held, FEATURE_NAMES, rule, INTERVENTION)
        changes = [(result['selection_aggregate'][d]['conditions'][c]['balanced_accuracy_at_zero'],
                    v['balanced_accuracy_at_zero'])
                   for d, z in encoded.items() for c, v in z['conditions'].items() if c != 'original']
        result['criteria'].update({
            'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
            'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
        })
        result['gate_passed'] = all(result['criteria'].values())
        result['held_encoding_aggregate'] = encoded
        result['fit_diagnostics'] = fit
        result['selection_paired_drift'] = {}
        raw = [r for r in rows if r['variant'] == 'raw']
        raw_selected = [r for r in raw if r['role'] == 'selection']
        counts = {d: sum(r['domain'] == d for r in raw_selected) for d in ('rr', 'chimera')}
        variance_weights = np.array([.5/counts[r['domain']] for r in raw_selected])
        raw_scores = rule.score([[float(r[n]) for n in FEATURE_NAMES] for r in raw_selected])
        variance = float(np.sum(variance_weights*(raw_scores-np.sum(variance_weights*raw_scores))**2))
        result['selection_raw_score_variance'] = variance
        result['normalized_selection_paired_drift'] = {}
        for variant, data in [('raw', rows), ('jpeg90_444_after_resize256', rows), (INTERVENTION, [*raw, *held])]:
            variants = ('raw',) if variant == 'raw' else ('raw', variant)
            delta, weights = paired_deltas(data, 'selection', variants)
            drift = np.sum((delta/np.array(rule.scale))*rule.weights, axis=1)
            result['selection_paired_drift'][variant] = float(np.sum(weights*drift**2))
            result['normalized_selection_paired_drift'][variant] = (
                result['selection_paired_drift'][variant]/variance if variance > 1e-12 else None)
            selected = [r for r in data if r['role'] == 'selection' and r['variant'] == variant]
            values = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in selected])
            batch = rule.score(values); single = np.array([rule.score([v])[0] for v in values])
            if not np.array_equal(batch, single):
                raise ValueError('Stable single/batch scores differ')
        output[mode] = result
        if not shuffled: rules[mode] = rule
        print(json.dumps({'candidate': mode, 'gate_passed': result['gate_passed'], 'fit': fit}), flush=True)
    passing = [m for m in rules if output[m]['gate_passed']]
    refused = output['source_shuffled_null']['gate_passed']
    chosen = max(passing, key=lambda m: output[m]['worst_processed_auc']) if passing and not refused else None
    return {'candidates': output, 'chosen': chosen, 'interpretation_refused': refused,
            'single_batch_scores_exact': True, 'scope': 'Repeated RR/Chimera development, no blind test'}, rules


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    path = OUTPUT/('pilot.json' if args.pilot else 'iteration.json')
    if path.exists(): raise FileExistsError('Preserve eighth calculation')
    rows, held, receipt = load_inputs()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    start = perf_counter()
    with threadpool_limits(limits=1):
        if args.pilot:
            _, fit = fit_rule(rows, 0., config, receipt['inventory_sha256'])
            result = {'fit': fit, 'pilot_candidate': 'lambda0', 'target_fits': 4,
                      'scope': 'Full-width single fit timing; screen/bootstrap not included'}
        else:
            result, rules = screen(rows, held, config, receipt['inventory_sha256'])
            for mode, rule in rules.items(): rule.save(OUTPUT/f'{mode}_rule.json')
            result['rule_files'] = {m: file_sha256(OUTPUT/f'{m}_rule.json') for m in rules}
            if result['chosen']:
                old = WORK_DIR/'robust_statistics/rr_first_iteration/benchmark.csv'
                receipt_old = json.loads(old.with_suffix('.json').read_text(encoding='utf-8'))
                if file_sha256(old) != receipt_old['csv_sha256']:
                    raise ValueError('Frozen benchmark selection changed')
                names = {'rr/'+r['filename'] for r in read_rows(old)}
                selected = [r for r in rows if r['variant'] == 'raw' and r['filename'] in names]
                if len(selected) != 60: raise ValueError('Benchmark denominator differs')
                rule = rules[result['chosen']]
                items = [{'filename': r['filename'], 'path': image_path(r), 'sha256': r['sha256'],
                          'expected_score': float(rule.score([[float(r[n]) for n in FEATURE_NAMES]])[0])} for r in selected]
                write_json(OUTPUT/'benchmark.json', benchmark_files(StableDetector(rule), items))
    result.update({'elapsed_s': perf_counter()-start, 'code_pins': code_pins(),
                   'parent_features_receipt_sha256': file_sha256(PARENT/'features.json'),
                   'parent_iteration_sha256': file_sha256(PARENT/'iteration.json')})
    write_json(path, result)


if __name__ == '__main__': main()
