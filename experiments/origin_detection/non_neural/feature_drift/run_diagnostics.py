"""Fit-only hand-descriptor diagnostics; never evaluates a neural representation."""

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np

from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES as RESIDUAL
from palimpsest.detection.algorithms.wavelet_envelopes import FEATURE_NAMES as ENVELOPE
from palimpsest.detection.algorithms.color_relations import FEATURE_NAMES as COLOR
from palimpsest.evaluation.feature_drift import paired_descriptor_drift, paired_indices
from palimpsest.evaluation.features import IDENTITY_FIELDS
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR

OUTPUT = WORK_DIR/'non_neural/feature_drift'
NAMES = RESIDUAL+ENVELOPE+COLOR
VARIANT = 'jpeg90_444_after_resize256'
BLOCKS = {'ordinal': (0, 28), 'residual': (28, 379), 'envelope': (379, 499),
    'color_order': (499, 634), 'color_ties': (634, 661), 'color_joint': (661, 850)}
PARTS = (
    ('joint_residual_statistics', RESIDUAL, '8e60a5bb4bbb5a2c662fdb86dbc95acd2a4fc71ce487ce2e28d12eba8a319fd4'),
    ('wavelet_envelopes', ENVELOPE, 'f34df1b22278606791d4a1c09a71dd6adecf8301dd1b3766df41426316b4d6b0'),
    ('color_relations', COLOR, '0a00b620f1a2bdb2d130fd1026565a4b4475ac987f919fe8845d30f59f3d9dc5'),
)


def code_pins():
    files = [Path(__file__), Path(__file__).with_name('README.md'),
        REPO_ROOT/'src/palimpsest/evaluation/feature_drift.py',
        REPO_ROOT/'tests/evaluation/test_feature_drift.py']
    return {p.relative_to(REPO_ROOT).as_posix(): file_sha256(p) for p in files}


def write_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def load_fit_features():
    tables = []; receipts = {}
    for name, names, expected in PARTS:
        folder = WORK_DIR/'robust_statistics'/name
        receipt_path = folder/'features.json'
        if file_sha256(receipt_path) != expected:
            raise ValueError(f'Historical descriptor receipt changed: {name}')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        path = folder/'features.csv'
        if file_sha256(path) != receipt['csv_sha256']:
            raise ValueError(f'Historical descriptor CSV changed: {name}')
        table = {}; count = 0
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if not set(names) <= set(reader.fieldnames):
                raise ValueError(f'Incomplete descriptor columns: {name}')
            for row in reader:
                count += 1
                if row['role'] != 'fit':
                    continue  # Other roles are neither converted to numeric features nor analyzed.
                if row['variant'] not in ('raw', VARIANT):
                    raise ValueError('Unexpected fit encoding support')
                key = row['filename'], row['variant']
                if key in table:
                    raise ValueError('Duplicate fit feature identity')
                metadata = {field: row[field] for field in (*IDENTITY_FIELDS, 'variant')}
                table[key] = metadata, np.array([float(row[n]) for n in names])
        if count != receipt['records'] or len(table) != 7560:
            raise ValueError('Fit descriptor coverage differs')
        tables.append(table)
        receipts[name] = {'receipt_sha256': expected, 'csv_sha256': receipt['csv_sha256'],
            'historical_code_pins': receipt['code_pins'], 'inventory_sha256': receipt['inventory_sha256']}
    if len({r['inventory_sha256'] for r in receipts.values()}) != 1:
        raise ValueError('Historical source rosters differ')
    keys = sorted(tables[0])
    if any(set(t) != set(keys) for t in tables):
        raise ValueError('Hand-feature parts have different query identities')
    rows, vectors = [], []
    for key in keys:
        metadata = tables[0][key][0]
        if any(t[key][0] != metadata for t in tables[1:]):
            raise ValueError('Hand-feature metadata parts conflict')
        rows.append(metadata)
        vectors.append(np.concatenate([t[key][1] for t in tables]))
    values = np.asarray(vectors)
    if values.shape != (7560, 850) or not np.isfinite(values).all():
        raise ValueError('Invalid combined hand-descriptor matrix')
    for domain, expected in (('rr', 540), ('chimera', 720)):
        sources = {r['src'] for r in rows if r['domain'] == domain}
        if len(sources) != expected:
            raise ValueError('Frozen fit source count changed')
        for source in sources:
            selected = [r for r in rows if r['domain'] == domain and r['src'] == source]
            if (len(selected) != 6 or len({r['condition'] for r in selected}) != 3
                    or len({(r['scene'], r['label']) for r in selected}) != 1):
                raise ValueError('Incomplete or conflicting fit source views')
    return rows, values, receipts


def diagnose(rows, values):
    results = []
    for domain in ('rr', 'chimera'):
        original = np.array([i for i, r in enumerate(rows) if r['domain'] == domain
            and r['condition'] == 'original' and r['variant'] == 'raw'])
        scale = np.std(values[original], axis=0)
        conditions = sorted({r['condition'] for r in rows if r['domain'] == domain})
        pairs = [('original', 'raw', c, 'raw') for c in conditions if c != 'original']
        pairs += [(c, 'raw', c, VARIANT) for c in conditions]
        for before, variant1, after, variant2 in pairs:
            a, b, y = paired_indices(rows, domain, before, variant1, after, variant2)
            for name, (start, stop) in BLOCKS.items():
                stats = paired_descriptor_drift(values[a, start:stop], values[b, start:stop],
                    y, scale[start:stop])
                results.append({'domain': domain, 'block': name, 'before': before+'/'+variant1,
                    'after': after+'/'+variant2, **stats})
    if len(results) != 60:
        raise ValueError('Expected sixty complete descriptor comparisons')
    return results


def pilot_rows(rows, values):
    chosen = set(); rng = random.Random(20261011)
    for domain in ('rr', 'chimera'):
        for label in ('REAL', 'FAKE'):
            sources = sorted({r['src'] for r in rows if r['domain'] == domain and r['label'] == label})
            rng.shuffle(sources)
            chosen.update((domain, s) for s in sources[:40])
    mask = [i for i, r in enumerate(rows) if (r['domain'], r['src']) in chosen]
    return [rows[i] for i in mask], values[mask]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--record-controls', action='store_true')
    group.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pins = code_pins()
    if args.record_controls:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/evaluation/test_feature_drift.py'],
            cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Descriptor plants failed: '+result.stdout+result.stderr)
        write_new(OUTPUT/'controls.json', {'passed': True, 'stdout': result.stdout, 'code_pins': pins,
            'scope': 'Analytic descriptor controls and metadata refusals; no scientific independence claim'})
        print(result.stdout, end='')
        return
    controls = json.loads((OUTPUT/'controls.json').read_text(encoding='utf-8'))
    if not controls['passed'] or controls['code_pins'] != pins:
        raise ValueError('Descriptor controls changed or failed')
    output_path = OUTPUT/('pilot.json' if args.pilot else 'diagnostics.json')
    if output_path.exists():
        raise FileExistsError('Preserve descriptor diagnostics')
    start = perf_counter()
    rows, values, receipts = load_fit_features()
    ready = perf_counter()
    if args.pilot:
        small_rows, small_values = pilot_rows(rows, values)
        times = []
        for _ in range(2):
            begin = perf_counter(); diagnose(small_rows, small_values); times.append(perf_counter()-begin)
        write_new(output_path, {'passed': True, 'sources': 160, 'records': len(small_rows),
            'times_s': times, 'input_load_s': ready-start, 'projected_full_statistics_s': times[-1]*1260/160,
            'assumption': 'Same configuration; statistics linear in source count; input cost recorded separately',
            'code_pins': pins, 'historical_receipts': receipts})
        print(json.dumps({'pilot_s': times, 'projected_full_statistics_s': times[-1]*1260/160}))
        return
    pilot = json.loads((OUTPUT/'pilot.json').read_text(encoding='utf-8'))
    if (not pilot['passed'] or pilot['code_pins'] != pins or pilot['historical_receipts'] != receipts
            or pilot['projected_full_statistics_s'] > 7200):
        raise ValueError('Cost pilot changed, failed or requires restructuring')
    table = diagnose(rows, values)
    write_new(output_path, {'comparisons': table, 'fit_sources': 1260, 'records': len(rows),
        'dimensions': len(NAMES), 'threshold_or_selection_analyzed': 0,
        'input_load_s': ready-start, 'statistics_s': perf_counter()-ready, 'code_pins': pins,
        'historical_receipts': receipts, 'scope': 'Fit-only exposed-development descriptor diagnostics; not BA or independent validation'})
    print(json.dumps({'comparisons': len(table), 'fit_sources': 1260, 'records': len(rows),
        'input_load_s': ready-start, 'statistics_s': perf_counter()-ready}))


if __name__ == '__main__':
    main()
