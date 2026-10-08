"""Stream existing query/perturbation features, never matching source images."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from time import perf_counter

import numpy as np

from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES as GLOBAL_NAMES
from palimpsest.detection.representations.intermediate_clip import MID_NAMES
from palimpsest.detection.representations.query_sensitivity import shifted_cosine_drift
from palimpsest.evaluation.features import IDENTITY_FIELDS, validate_feature_cache
from palimpsest.io.hashing import file_sha256
from experiments.origin_detection.query_sensitivity.protocol import (
    PARENT, OUTPUT, NAMES, QUERY_VARIANT, parent_receipt, code_pins, write_json)


def selected(row, stride):
    return not stride or (row['role'] == 'fit'
        and int(sha256(row['filename'].encode()).hexdigest(), 16) % stride == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-stride', type=int, default=0)
    args = parser.parse_args()
    if args.pilot_stride < 0:
        parser.error('Stride must be nonnegative')
    parent = parent_receipt()
    directory = OUTPUT.with_name(f'query_sensitivity_pilot_{args.pilot_stride}') if args.pilot_stride else OUTPUT
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory/name).exists() for name in ('features.csv', 'features.json')):
        raise FileExistsError('Preserve sensitivity materialization')
    if not args.pilot_stride:
        for stride in (32, 16):
            old = OUTPUT.with_name(f'query_sensitivity_pilot_{stride}')/'features.json'
            pilot = json.loads(old.read_text(encoding='utf-8'))
            if not pilot['passed'] or pilot['code_pins'] != code_pins():
                raise ValueError('Preparation pilot changed')
    start = perf_counter()
    pending, seen, inventory, output = {}, set(), [], []
    times, scanned = [], 0
    with (PARENT/'features.csv').open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        metadata_names = [name for name in reader.fieldnames
            if name not in (*GLOBAL_NAMES, *MID_NAMES, 'preprocess_ms', 'statistics_ms')]
        for row in reader:
            scanned += 1
            identity = row['filename'], row['variant']
            if identity in seen:
                raise ValueError('Duplicate parent identity')
            seen.add(identity)
            if row['variant'] not in ('raw', QUERY_VARIANT) or not selected(row, args.pilot_stride):
                continue
            values = np.array([float(row[name]) for name in (*GLOBAL_NAMES, *MID_NAMES)])
            if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
                raise ValueError('Invalid parent embeddings')
            previous = pending.pop(row['filename'], None)
            if previous is None:
                pending[row['filename']] = row, values
                continue
            other, other_values = previous
            if (row['variant'] == other['variant']
                    or any(row[key] != other[key] for key in IDENTITY_FIELDS)):
                raise ValueError('Perturbation is not the same native query')
            raw = row if row['variant'] == 'raw' else other
            t = perf_counter()
            drift = [shifted_cosine_drift(values[:768], other_values[:768]),
                     shifted_cosine_drift(values[768:], other_values[768:])]
            times.append(perf_counter()-t)
            meta = {name: raw[name] for name in metadata_names}
            meta['variant'] = 'raw'
            inventory.append(meta)
            output.append({**meta, **dict(zip(NAMES, drift))})
    if pending or scanned != 20160:
        raise ValueError('Incomplete parent traversal/query pairs')
    validate_feature_cache(output, inventory, NAMES, bounds=(0, 2))
    if not args.pilot_stride and len(output) != 6300:
        raise ValueError('Missing native queries')
    counts = Counter((r['domain'], r['scene'], r['condition'], r['label']) for r in output)
    if len(counts) != 24 or any(count < 1 for count in counts.values()):
        raise ValueError('Pilot/full strata missing')
    path = directory/'features.csv'
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[*metadata_names, *NAMES])
        writer.writeheader()
        writer.writerows(output)
    elapsed = perf_counter()-start
    write_json(directory/'features.json', {'passed': True, 'records': len(output),
        'scanned_parent_rows': scanned, 'elapsed_s': elapsed, 'drift_compute_s': sum(times),
        'first_half_median_ms': 1000*float(np.median(times[:len(times)//2])),
        'second_half_median_ms': 1000*float(np.median(times[len(times)//2:])),
        'pilot_stride': args.pilot_stride, 'csv_sha256': file_sha256(path),
        'parent_receipt_sha256': file_sha256(PARENT/'features.json'),
        'parent_csv_sha256': parent['csv_sha256'], 'inventory_sha256': parent['inventory_sha256'],
        'code_pins': code_pins(), 'strata': {'/'.join(k): n for k, n in sorted(counts.items())},
        'scope': 'Stream cached embeddings;no image decoding/encoder/query generation;not inference latency'})
    print({'records': len(output), 'elapsed_s': elapsed, 'drift_compute_s': sum(times)}, flush=True)


if __name__ == '__main__':
    main()
