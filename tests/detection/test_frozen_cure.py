"""Preserve unbounded released vectors; refuse the old default bounded cache."""

import numpy as np
import pytest

from palimpsest.detection.representations.frozen_cure import FEATURE_NAMES, pack_features
from palimpsest.evaluation.features import validate_feature_cache


def test_signed_raw_storage_and_cache_bounds():
    full = np.arange(1024, dtype=np.float32) / 64 - 8
    projected = np.linspace(-12, 12, 128).astype(np.float16)
    values = pack_features(full, projected)
    assert values[0] == -8 and values[1023] == 7.984375
    assert values[1024] == -12 and values[-1] == 12
    assert np.array_equal(values[:1024].astype(np.float32), full)
    assert np.array_equal(values[1024:].astype(np.float16), projected)
    record = dict(filename='known', src='s', condition='original', role='fit', label='REAL',
                  domain='toy', scene='known', sha256='fixed')
    row = {**record, 'variant': 'raw', **dict(zip(FEATURE_NAMES, values))}
    validate_feature_cache([row], [record], FEATURE_NAMES, bounds=(-np.inf, np.inf))
    with pytest.raises(ValueError, match='Invalid feature values'):
        validate_feature_cache([row], [record], FEATURE_NAMES)


def test_invalid_raw_vectors_refused():
    with pytest.raises(ValueError):
        pack_features(np.zeros(1023), np.zeros(128))
    full = np.zeros(1024)
    full[0] = np.nan
    with pytest.raises(ValueError):
        pack_features(full, np.zeros(128))
