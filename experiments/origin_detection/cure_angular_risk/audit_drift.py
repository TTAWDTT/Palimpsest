"""Fit-only same-source radial drift description; never selects a candidate."""

from collections import defaultdict

import numpy as np

from palimpsest.detection.representations.angular_features import paired_drift
from palimpsest.detection.representations.frozen_cure import FULL_NAMES
from palimpsest.io.hashing import file_sha256
from experiments.origin_detection.cure_angular_risk.run_iteration import OUTPUT, code_pins, controls_gate
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs
from experiments.origin_detection.phase_statistics.run_iteration import write_json


def collect_pairs(rows):
    indexed = {}
    for row in rows:
        if row['role'] != 'fit':
            continue
        key = (row['domain'], row['scene'], row['src'], row['condition'], row['variant'])
        if key in indexed:
            raise ValueError('Duplicate drift pair key')
        indexed[key] = row
    grouped = defaultdict(list)
    for (domain, scene, src, condition, variant), row in indexed.items():
        reference = indexed.get((domain, scene, src, 'original', 'raw'))
        if reference is None or row['label'] != reference['label']:
            raise ValueError('Missing/conflicting drift reference')
        grouped[domain+'/'+condition+'/'+variant+'/'+row['label']].append((reference, row))
    return grouped


def main():
    destination = OUTPUT/'fit_drift.json'
    if destination.exists():
        raise FileExistsError('Preserve fit drift audit')
    controls = controls_gate()
    # Key assembly corruption is checked before accessing cached real features.
    plant = {'role': 'fit', 'domain': 'toy', 'scene': 'toy', 'src': '0',
             'condition': 'original', 'variant': 'raw', 'label': 'REAL'}
    if sum(map(len, collect_pairs([plant]).values())) != 1:
        raise ValueError('Known drift assembly failed')
    for broken in ([plant, plant], [plant, {**plant, 'condition': 'processed', 'label': 'FAKE'}]):
        try:
            collect_pairs(broken)
        except ValueError:
            pass
        else:
            raise ValueError('Corrupted drift key/reference accepted')
    rows, parent = cure_inputs()
    grouped = collect_pairs(rows)
    if sum(map(len, grouped.values())) != 7560:
        raise ValueError('Fit-only drift coverage changed')
    results = {}
    for key, pairs in sorted(grouped.items()):
        a = [[float(r[n]) for n in FULL_NAMES] for r, _ in pairs]
        b = [[float(r[n]) for n in FULL_NAMES] for _, r in pairs]
        result = paired_drift(a, b)
        results[key] = {'pairs': len(pairs), **{k: dict(zip(('p05', 'p50', 'p95'),
                       np.quantile(v, [.05, .5, .95]).tolist())) for k, v in result.items()}}
    write_json(destination, {'fit_records': 7560, 'groups': results, 'code_pins': code_pins(),
        'software_controls_sha256': file_sha256(controls), 'inventory_sha256': parent['inventory_sha256'],
        'key_controls': {'known_clean_passed': True, 'duplicate_refused': True, 'conflicting_label_refused': True},
        'scope': 'Fit-only feature decomposition;does not establish angular invariance in unseen physical channels'})


if __name__ == '__main__':
    main()
