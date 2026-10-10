"""Seventh development iteration; reuse signed features, with eval-only JPEG70."""

import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
import random
from time import perf_counter
import tomllib

import cv2
import numpy as np
from PIL import Image

from palimpsest.detection.algorithms.forest import ForestRule
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES, extract_features, resize256
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.residual_statistics.run_iteration import (
    OUTPUT as PARENT, aggregate, benchmark, screen_rule, signed_features,
)
from .fit_rules import MODES, fit_rule

CONFIG = REPO_ROOT / 'configs/evaluation/residual_training.toml'
OUTPUT = WORK_DIR / 'robust_statistics/residual_training'
INTERVENTION = 'jpeg70_420_after_resize256'


def code_pins():
    directory = REPO_ROOT / 'experiments/origin_detection/residual_training'
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p)
            for p in sorted([CONFIG, directory / 'README.md', *directory.glob('*.py')])}


def selection_inventory(rows):
    return [{k: v for k, v in r.items() if k not in FEATURE_NAMES and k not in
             ('variant', 'preprocess_ms', 'statistics_ms')}
            for r in rows if r['role'] == 'selection' and r['variant'] == 'raw']


def audit_intervention(rows, inventory):
    validate_feature_cache(rows, inventory, FEATURE_NAMES, variants=(INTERVENTION,), bounds=(0, 1))
    for row in rows:
        digest = row['jpeg_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid intervention byte SHA')
        f = np.array([float(row[n]) for n in FEATURE_NAMES])
        if (any(not np.isclose(f[i:i+6].sum(), 1, atol=1e-12, rtol=0) for i in range(0, 28, 7))
                or any(not np.isclose(f[i:i+39].sum(), 1, atol=1e-12, rtol=0) for i in range(28, 379, 39))):
            raise ValueError('Intervention histogram not normalized')
    return rows


def extract_intervention(inventory, directory):
    start = perf_counter()
    records = []
    for index, row in enumerate(inventory, 1):
        path = image_path(row)
        if row['role'] != 'selection' or file_sha256(path) != row['sha256']:
            raise ValueError('Intervention role/source SHA changed')
        with Image.open(path) as image:
            if image.size != (int(row['width']), int(row['height'])):
                raise ValueError('Intervention source dimensions changed')
            rgb = np.asarray(image.convert('RGB'))
        stream = BytesIO()
        Image.fromarray(resize256(rgb)).save(stream, format='JPEG', quality=70, subsampling=2)
        byte_sha = sha256(stream.getvalue()).hexdigest()
        stream.seek(0)
        with Image.open(stream) as image:
            pixels = np.asarray(image.convert('RGB'))
        f = extract_features(pixels)
        records.append({**row, 'variant': INTERVENTION, 'jpeg_sha256': byte_sha,
                        **dict(zip(FEATURE_NAMES, f.values.tolist()))})
        if index % 60 == 0 or index == len(inventory):
            print(json.dumps({'images': index, 'total': len(inventory), 'elapsed_s': perf_counter()-start}), flush=True)
    audit_intervention(records, inventory)
    with (directory / 'features.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    return {'images': len(inventory), 'elapsed_s': perf_counter()-start,
            'csv_sha256': file_sha256(directory / 'features.csv'), 'code_pins': code_pins(),
            'parent_features_receipt_sha256': file_sha256(PARENT / 'features.json')}


def screen(rows, intervention, config, manifest_sha):
    output, rules = {}, {}
    for mode in (*MODES, 'source_shuffled_null'):
        shuffled = mode == 'source_shuffled_null'
        rule, training = fit_rule(rows, config['null_mode'] if shuffled else mode,
                                  seed=config['seed'], manifest_sha=manifest_sha, shuffled=shuffled)
        if mode == 'equal_view_raw':
            parent = json.loads((PARENT / 'iteration.json').read_text(encoding='utf-8'))
            path = PARENT / 'hybrid_rule.json'
            if file_sha256(path) != parent['rule_files']['hybrid'] or rule.payload() != ForestRule.load(path).payload():
                raise ValueError('Sixth control was not exactly reproduced')
        result = screen_rule(rows, FEATURE_NAMES, rule, config)
        held = aggregate(intervention, FEATURE_NAMES, rule, INTERVENTION)
        changes = [(result['selection_aggregate'][d]['conditions'][c]['balanced_accuracy_at_zero'],
                    v['balanced_accuracy_at_zero'])
                   for d, z in held.items() for c, v in z['conditions'].items() if c != 'original']
        result['criteria'].update({
            'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
            'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
        })
        result['gate_passed'] = all(result['criteria'].values())
        result['held_encoding_aggregate'] = held
        result['training'] = training
        audits = {}
        for variant, data in [('raw', rows), ('jpeg90_444_after_resize256', rows), (INTERVENTION, intervention)]:
            selected = [r for r in data if r['role'] == 'selection' and r['variant'] == variant]
            x = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in selected])
            batch = rule.score(x)
            single = np.array([rule.score([v])[0] for v in x])
            differences = int(np.count_nonzero((batch > rule.threshold) != (single > rule.threshold)))
            if differences:
                raise ValueError('Single/batch decisions differ')
            audits[variant] = {'records': len(x), 'maximum_score_difference': float(np.max(np.abs(batch-single))),
                               'decision_differences': differences}
        result['single_batch_audit'] = audits
        output[mode] = result
        if not shuffled:
            rules[mode] = rule
        print(json.dumps({'candidate': mode, 'gate_passed': result['gate_passed']}), flush=True)
    passing = [m for m in MODES if output[m]['gate_passed']]
    refused = output['source_shuffled_null']['gate_passed']
    chosen = max(passing, key=lambda m: output[m]['worst_processed_auc']) if passing and not refused else None
    return {'candidates': output, 'chosen': chosen, 'interpretation_refused': refused,
            'scope': 'Repeated development sources; eval-only encoding is not an independent dataset test'}, rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', type=int, default=0)
    group.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.pilot < 0:
        parser.error('Pilot must be positive')
    cv2.setNumThreads(1)
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    rows, parent_receipt = signed_features()
    inventory = selection_inventory(rows)
    if len(inventory) != 1260:
        raise ValueError('Frozen selection denominator differs')
    if args.pilot or args.prepare:
        directory = OUTPUT.with_name('residual_training_pilot') if args.pilot else OUTPUT
        if directory.exists():
            raise FileExistsError('Preserve seventh extraction')
        if args.pilot:
            random.Random(config['seed']).shuffle(inventory)
            inventory = inventory[:args.pilot]
        directory.mkdir(parents=True)
        write_json(directory / 'features.json', extract_intervention(inventory, directory))
        return
    receipt = json.loads((OUTPUT / 'features.json').read_text(encoding='utf-8'))
    if (receipt['code_pins'] != code_pins() or receipt['csv_sha256'] != file_sha256(OUTPUT / 'features.csv')
            or receipt['parent_features_receipt_sha256'] != file_sha256(PARENT / 'features.json')):
        raise ValueError('Frozen seventh lineage changed')
    intervention = audit_intervention(read_rows(OUTPUT / 'features.csv'), inventory)
    if (OUTPUT / 'iteration.json').exists():
        raise FileExistsError('Preserve seventh evaluation')
    start = perf_counter()
    result, rules = screen(rows, intervention, config, parent_receipt['inventory_sha256'])
    for mode, rule in rules.items():
        rule.save(OUTPUT / f'{mode}_rule.json')
    result['rule_files'] = {m: file_sha256(OUTPUT / f'{m}_rule.json') for m in rules}
    result.update({'code_pins': code_pins(), 'features_receipt_sha256': file_sha256(OUTPUT / 'features.json'),
                   'parent_iteration_sha256': file_sha256(PARENT / 'iteration.json'), 'elapsed_s': perf_counter()-start})
    write_json(OUTPUT / 'iteration.json', result)
    if result['chosen']:
        write_json(OUTPUT / 'benchmark.json', benchmark(rows, {result['chosen']: rules[result['chosen']]}))


if __name__ == '__main__':
    main()
