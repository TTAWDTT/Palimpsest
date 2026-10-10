"""Signed old cache plus missing development Q60 views; roles never reassigned."""

import numpy as np

from palimpsest.evaluation.numeric_features import numeric_rows
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.source_panel_crossfit import SourcePanel, panel_calibration_views, wrong_panel_sources
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from experiments.origin_detection.strong_token_views.prepare_features import (
    OUTPUT as INCREMENT, PARENT, parent_data, validate, VARIANT)
from experiments.origin_detection.cure_token_covariance.prepare_features import FEATURE_NAMES, audit_receipt
from experiments.origin_detection.token_csp_consistency.run_iteration import NAMES, CONFIG, VARIANTS

PANEL = SourcePanel((VARIANTS[0], VARIANTS[1], VARIANTS[3]))


def inputs():
    metadata, values, inventory, parent = parent_data()
    increment = audit_receipt(INCREMENT/'features.json')
    if (increment['records'] != 5040 or increment['variant'] != VARIANT or increment['complete_reference_exact']
            or increment['parent_receipt_sha256'] != file_sha256(PARENT/'features.json')
            or increment['inventory_sha256'] != parent['inventory_sha256']
            or increment['encoder'] != parent['encoder'] or increment['runtime'] != parent['runtime']
            or file_sha256(INCREMENT/'vectors.npy') != increment['vectors_sha256']
            or file_sha256(INCREMENT/'metadata.csv') != increment['metadata_sha256']):
        raise ValueError('Incremental support changed or incorrectly claims old full reference')
    extra_metadata = read_rows(INCREMENT/'metadata.csv'); extra_values = np.load(INCREMENT/'vectors.npy', mmap_mode='r')
    expected = [r for r in inventory if r['role'] in ('fit', 'threshold')]
    validate(extra_metadata, extra_values, expected)
    if tuple(FEATURE_NAMES[1024:]) != NAMES: raise ValueError('Mixed covariance column order differs')
    rows = numeric_rows(metadata, values[:, 1024:], NAMES) + numeric_rows(extra_metadata, extra_values[:, 1024:], NAMES)
    if len(rows) != 20160: raise ValueError('Expanded numeric count differs')
    for role, count, variants in (('fit', 11340, PANEL.variants), ('threshold', 3780, PANEL.variants),
                                  ('selection', 5040, VARIANTS)):
        selected = [r for r in rows if r['role'] == role]
        if len(selected) != count: raise ValueError('Expanded role count differs')
        validate_feature_cache(selected, [r for r in inventory if r['role'] == role], NAMES,
                               variants=variants, bounds=(-np.inf, np.inf))
    return rows, parent


def training(rows):
    records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in PANEL.variants],
                     key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    x, y, w, s = weighted_source_arrays(rows, NAMES, PANEL.variants,
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    PANEL.check(records)
    if len(x) != 11340: raise ValueError('Expanded training rows differ')
    return records, x, y, w, s


def calibration_views(records, x, labels, mask):
    return panel_calibration_views(records, x, labels, mask, NAMES, PANEL)


def wrong_sources(rows, sources):
    records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in PANEL.variants],
                     key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    return wrong_panel_sources(records, sources, seed=CONFIG['seed'])
