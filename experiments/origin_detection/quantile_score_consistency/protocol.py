"""Signed old prefix and complete new token-shape numerical cache."""

import json

import numpy as np

from palimpsest.detection.representations.frozen_cure_quantiles import FEATURE_NAMES as NAMES
from palimpsest.evaluation.features import IDENTITY_FIELDS, validate_feature_cache
from palimpsest.evaluation.numeric_features import numeric_rows
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.source_panel_crossfit import panel_calibration_views, wrong_panel_sources
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT
from experiments.origin_detection.cure_token_quantiles_full.prepare_features import OUTPUT as FULL
from experiments.origin_detection.semantic_score_margin.protocol import (
    inputs as old_inputs, CONFIG, PANEL, VARIANTS as VARIANTS)


def inputs():
    path = FULL/'features.json'
    if not path.is_file():
        raise ValueError('Complete quantile cache not yet signed')
    receipt = json.loads(path.read_text())
    if not receipt['passed'] or receipt['records'] != 20160 or receipt['dimensions'] != len(NAMES):
        raise ValueError('Full quantile receipt failed')
    for name, expected in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/name) != expected:
            raise ValueError('Quantile extraction source changed')
    rows = []
    for part, count in (('base', 15120), ('q60', 5040)):
        directory = FULL/part
        sub = json.loads((directory/'features.json').read_text())
        for filename, key in (('vectors.npy', 'vectors_sha256'), ('metadata.csv', 'metadata_sha256')):
            if file_sha256(directory/filename) != sub[key] or sub[key] != receipt['parts'][part][key]:
                raise ValueError('Quantile subcache changed')
        if sub['records'] != count:
            raise ValueError('Quantile subcache count differs')
        if part == 'base' and not sub['all_parent_prefix_exact']:
            raise ValueError('Legacy prefix not signed')
        if part == 'q60' and not sub['all_parent_prefix_exact_in_final_validation']:
            raise ValueError('Q60 old prefix/query gate absent')
        rows.extend(numeric_rows(read_rows(directory/'metadata.csv'),
            np.load(directory/'vectors.npy', mmap_mode='r'), NAMES))
    old, parent = old_inputs()
    lookup = {(r['filename'], r['variant']): r for r in old}
    if len(lookup) != 20160 or len(rows) != 20160 or parent['inventory_sha256'] != receipt['inventory_sha256']:
        raise ValueError('Legacy/new source roster differs')
    seen = set()
    for row in rows:
        key = row['filename'], row['variant']
        if key in seen or key not in lookup:
            raise ValueError('Duplicate/missing quantile view')
        seen.add(key); before = lookup[key]
        if (any(row[field] != before[field] for field in IDENTITY_FIELDS)
                or not np.array_equal(row.vector[:3600], before.vector)
                or float(row['probability_fake']) != float(before['probability_fake'])):
            raise ValueError('Quantile parent prefix/probability/identity differs')
    if seen != set(lookup):
        raise ValueError('Missing quantile identity')
    inventory = [r.metadata for r in rows if r['variant'] == 'raw']
    for role, count, variants in (('fit', 11340, PANEL.variants), ('threshold', 3780, PANEL.variants),
                                  ('selection', 5040, VARIANTS)):
        chosen = [r for r in rows if r['role'] == role]
        if len(chosen) != count:
            raise ValueError('Quantile source role count changed')
        validate_feature_cache(chosen, [r for r in inventory if r['role'] == role], NAMES,
            variants=variants, bounds=(-np.inf, np.inf))
    return rows, receipt


def training(rows):
    records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in PANEL.variants],
        key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    x, y, weights, sources = weighted_source_arrays(rows, NAMES, PANEL.variants,
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    PANEL.check(records)
    return records, x, y, weights, sources


def calibration_views(records, x, labels, mask):
    return panel_calibration_views(records, x, labels, mask, NAMES, PANEL)


def wrong_sources(rows, sources):
    records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in PANEL.variants],
        key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    return wrong_panel_sources(records, sources, seed=CONFIG['seed'])
