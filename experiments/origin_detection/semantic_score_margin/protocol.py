"""Recover the already stored raw prefix after validating the expanded cache."""

import numpy as np

from palimpsest.evaluation.numeric_features import numeric_rows
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.source_panel_crossfit import panel_calibration_views, wrong_panel_sources
from palimpsest.io.tables import read_rows
from experiments.origin_detection.strong_score_margin.protocol import (
    inputs as mixed_inputs, INCREMENT, PARENT, PANEL, CONFIG, VARIANTS as VARIANTS, FEATURE_NAMES)

NAMES = FEATURE_NAMES


def inputs():
    checked, parent = mixed_inputs()
    rows = []
    for directory in (PARENT, INCREMENT):
        metadata = read_rows(directory/'metadata.csv'); values = np.load(directory/'vectors.npy', mmap_mode='r')
        rows.extend(numeric_rows(metadata, values, NAMES))
    if len(rows) != len(checked) or len(rows) != 20160: raise ValueError('Full-prefix combined row count differs')
    for complete, mixed in zip(rows, checked):
        if complete.metadata != mixed.metadata or not np.array_equal(complete.vector[1024:], mixed.vector):
            raise ValueError('Full-prefix traversal or mixed columns differ')
    return rows, parent


def training(rows):
    records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in PANEL.variants],
                     key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    x, y, weights, sources = weighted_source_arrays(rows, NAMES, PANEL.variants,
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    PANEL.check(records)
    if len(x) != 11340: raise ValueError('Semantic complement training count differs')
    return records, x, y, weights, sources


def calibration_views(records, x, labels, mask):
    return panel_calibration_views(records, x, labels, mask, NAMES, PANEL)


def wrong_sources(rows, sources):
    records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in PANEL.variants],
                     key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    return wrong_panel_sources(records, sources, seed=CONFIG['seed'])
