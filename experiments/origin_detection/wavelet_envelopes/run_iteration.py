"""Signed fixed-envelope features and a preregistered four-representation screen."""

import argparse
import csv
from io import BytesIO
import json
from time import perf_counter
import random
import tomllib

import cv2
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.stable_rule import fit_stable_rule
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES as OLD_NAMES, resize256
from palimpsest.detection.algorithms.wavelet_envelopes import FIRST_NAMES, SECOND_NAMES, FEATURE_NAMES, extract_envelopes
from palimpsest.evaluation.features import feature_views, validate_feature_cache, IDENTITY_FIELDS
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.conditional_residual.run_iteration import final_target_screen
from experiments.origin_detection.paired_stability.fit_rules import paired_deltas
from experiments.origin_detection.paired_stability.run_iteration import load_inputs, INTERVENTION
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.residual_statistics.run_iteration import screen_rule, aggregate
from experiments.origin_detection.residual_training.fit_rules import training_arrays, TRAINING_VARIANT
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold

CONFIG = REPO_ROOT/'configs/evaluation/wavelet_envelopes.toml'
OUTPUT = WORK_DIR/'robust_statistics/wavelet_envelopes'
MODES = {'first': FIRST_NAMES, 'second': SECOND_NAMES, 'both': FEATURE_NAMES, 'hybrid': OLD_NAMES+FEATURE_NAMES}


def code_pins():
    directory = REPO_ROOT/'experiments/origin_detection/wavelet_envelopes'
    files = [CONFIG, REPO_ROOT/'src/palimpsest/detection/algorithms/wavelet_envelopes.py',
             REPO_ROOT/'tests/detection/test_wavelet_envelopes.py', directory/'README.md', *directory.glob('*.py'),
             REPO_ROOT/'src/palimpsest/detection/algorithms/paired_stability.py', REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/stable_rule.py',
             REPO_ROOT/'experiments/origin_detection/conditional_residual/run_iteration.py',
             REPO_ROOT/'experiments/origin_detection/threshold_calibration/fit_threshold.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def audit_new(rows, inventory):
    ordinary = [r for r in rows if r['variant'] != INTERVENTION]
    held = [r for r in rows if r['variant'] == INTERVENTION]
    validate_feature_cache(ordinary, inventory, FEATURE_NAMES, variants=('raw', TRAINING_VARIANT), bounds=(0, 1e6))
    selected = [r for r in inventory if r['role'] == 'selection']
    if selected:
        validate_feature_cache(held, selected, FEATURE_NAMES, variants=(INTERVENTION,), bounds=(0, 1e6))
    elif held:
        raise ValueError('Unexpected selection-only envelope features')
    return ordinary, held


def extract_inventory(inventory, directory):
    start = perf_counter(); rows = []
    for index, r in enumerate(inventory, 1):
        path = image_path(r)
        if file_sha256(path) != r['sha256']: raise ValueError('Envelope source SHA changed')
        with Image.open(path) as image:
            if image.size != (int(r['width']), int(r['height'])):
                raise ValueError('Envelope source dimensions changed')
            rgb = np.asarray(image.convert('RGB'))
        variants = ['raw', TRAINING_VARIANT]+([INTERVENTION] if r['role'] == 'selection' else [])
        for variant in variants:
            if variant == 'raw': pixels = rgb
            else:
                stream = BytesIO()
                Image.fromarray(resize256(rgb)).save(stream, format='JPEG',
                                                     quality=90 if variant == TRAINING_VARIANT else 70,
                                                     subsampling=0 if variant == TRAINING_VARIANT else 2)
                stream.seek(0)
                with Image.open(stream) as image: pixels = np.asarray(image.convert('RGB'))
            f = extract_envelopes(pixels)
            metadata = {k: v for k, v in r.items() if k not in OLD_NAMES and k not in ('variant', 'preprocess_ms', 'statistics_ms')}
            rows.append({**metadata, 'variant': variant, **dict(zip(FEATURE_NAMES, f.values.tolist())),
                         'preprocess_ms': f.preprocess_ms, 'statistics_ms': f.statistics_ms})
        if index % 60 == 0 or index == len(inventory):
            print(json.dumps({'images': index, 'total': len(inventory), 'records': len(rows),
                              'elapsed_s': round(perf_counter()-start, 3)}), flush=True)
    audit_new(rows, inventory)
    with (directory/'features.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return {'images': len(inventory), 'records': len(rows), 'elapsed_s': perf_counter()-start,
            'csv_sha256': file_sha256(directory/'features.csv'), 'code_pins': code_pins()}


def join_features(old, new):
    by_key = {(r['filename'], r['variant']): r for r in new}
    if len(by_key) != len(new) or set(by_key) != {(r['filename'], r['variant']) for r in old}:
        raise ValueError('Old/new feature coverage differs')
    result = []
    for r in old:
        n = by_key[r['filename'], r['variant']]
        if any(r[k] != n[k] for k in IDENTITY_FIELDS): raise ValueError('Old/new feature identity differs')
        result.append({**r, **{name: n[name] for name in FEATURE_NAMES}})
    return result


def fit_rule(rows, names, config, manifest_sha, *, shuffled=False):
    _, labels, weights, _ = training_arrays(rows, 'equal_domain_augmented', config['seed'], shuffled=shuffled)
    values = np.concatenate([x for variant in ('raw', TRAINING_VARIANT)
                             for _, x, _ in {**feature_views(rows, names, 'fit', processed=False, variant=variant),
                                              **feature_views(rows, names, 'fit', processed=True, variant=variant)}.values()])
    delta, pw = paired_deltas(rows, 'fit', ('raw', TRAINING_VARIANT), names)
    rule, diag = fit_stable_rule(values, labels, weights, delta, pw, feature_names=names, strength=config['strength'],
                                ridge=config['ridge'], scale_floor=config['scale_floor'], manifest_sha=manifest_sha)
    views = {k+'/'+variant: v for variant in ('raw', TRAINING_VARIANT) for processed in (False, True)
             for k, v in feature_views(rows, names, 'threshold', processed=processed, variant=variant).items()}
    rule, calibration = class_threshold(rule, views)
    return rule, {**diag, 'fit_records': len(values), 'paired_records': len(delta), 'feature_count': len(names),
                  'calibration': calibration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', type=int, default=0)
    group.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.pilot < 0: parser.error('Pilot must be positive')
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    cv2.setNumThreads(1)
    controls = OUTPUT/'controls.json'
    control = json.loads(controls.read_text(encoding='utf-8'))
    if control['returncode'] != 0 or control['test_sha256'] != file_sha256(REPO_ROOT/'tests/detection/test_wavelet_envelopes.py'):
        raise ValueError('Envelope planted controls failed or changed')
    old, old_held, receipt = load_inputs()
    inventory = [r for r in old if r['variant'] == 'raw']
    if args.prepare or args.pilot:
        directory = OUTPUT if args.prepare else OUTPUT.with_name('wavelet_envelopes_pilot')
        if (directory/'features.json').exists() or (directory/'features.csv').exists():
            raise FileExistsError('Preserve envelope extraction')
        directory.mkdir(parents=True, exist_ok=True)
        if args.pilot:
            random.Random(config['seed']).shuffle(inventory); inventory = inventory[:args.pilot]
        result = extract_inventory(inventory, directory)
        result.update({'inventory_sha256': receipt['inventory_sha256'], 'controls_sha256': file_sha256(controls),
                       'parent_features_receipt_sha256': file_sha256(WORK_DIR/'robust_statistics/joint_residual_statistics/features.json')})
        write_json(directory/'features.json', result)
        return
    if (OUTPUT/'iteration.json').exists(): raise FileExistsError('Preserve envelope evaluation')
    new_receipt = json.loads((OUTPUT/'features.json').read_text(encoding='utf-8'))
    if (new_receipt['code_pins'] != code_pins() or new_receipt['csv_sha256'] != file_sha256(OUTPUT/'features.csv')
            or new_receipt['inventory_sha256'] != receipt['inventory_sha256']):
        raise ValueError('Envelope code/cache/inventory changed')
    ordinary, held = audit_new(read_rows(OUTPUT/'features.csv'), inventory)
    rows, unseen = join_features(old, ordinary), join_features(old_held, held)
    start = perf_counter(); results = {}; rules = {}
    with threadpool_limits(limits=1):
        for mode in [*config['candidate_modes'], 'hybrid_source_shuffled_null']:
            shuffled = mode.endswith('_null'); names = MODES['hybrid' if shuffled else mode]
            rule, fit = fit_rule(rows, names, config, receipt['inventory_sha256'], shuffled=shuffled)
            value = screen_rule(rows, names, rule, config)
            encoded = aggregate(unseen, names, rule, INTERVENTION)
            changes = [(v['balanced_accuracy_at_zero'], encoded[d]['conditions'][c]['balanced_accuracy_at_zero'])
                       for d, z in value['selection_aggregate'].items() for c, v in z['conditions'].items() if c != 'original']
            value['criteria'].update({
                'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
                'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
                'both_encoded_class_accuracies': all(v[k] >= .55 for data in (value['recompression_aggregate'], encoded)
                                                     for z in data.values() for v in z['conditions'].values()
                                                     for k in ('real_accuracy_at_zero', 'fake_accuracy_at_zero'))})
            value.update({'gate_passed': all(value['criteria'].values()), 'held_encoding_aggregate': encoded,
                          'fit_diagnostics': fit, 'final_target': final_target_screen(value, encoded, config)})
            for variant, data in [('raw', rows), (TRAINING_VARIANT, rows), (INTERVENTION, unseen)]:
                selected = [r for r in data if r['role'] == 'selection' and r['variant'] == variant]
                x = np.array([[float(r[n]) for n in names] for r in selected])
                if not np.array_equal(rule.score(x), [rule.score([v])[0] for v in x]):
                    raise ValueError('Envelope single/batch score differs')
            results[mode] = value; rules[mode] = rule
            print(json.dumps({'candidate': mode, 'development_gate': value['gate_passed'], 'target': value['final_target']}), flush=True)
    refused = results['hybrid_source_shuffled_null']['criteria']['eight_auc_lower_bounds']
    passing = [k for k in config['candidate_modes'] if results[k]['gate_passed']]
    chosen = max(passing, key=lambda k: results[k]['worst_processed_auc']) if passing and not refused else None
    for k, rule in rules.items(): rule.save(OUTPUT/f'{k}_rule.json')
    write_json(OUTPUT/'iteration.json', {'candidates': results, 'development_chosen': chosen, 'interpretation_refused': refused,
                                       'single_batch_scores_exact': True, 'elapsed_s': perf_counter()-start,
                                       'features_receipt_sha256': file_sha256(OUTPUT/'features.json'), 'code_pins': code_pins(),
                                       'rule_files': {k: file_sha256(OUTPUT/f'{k}_rule.json') for k in rules},
                                       'scope': 'Four frozen representations on exposed development; no independent validation'})


if __name__ == '__main__': main()
