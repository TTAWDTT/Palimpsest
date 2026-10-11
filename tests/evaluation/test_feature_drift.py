"""Analytic plants and explicit refusal controls, not image performance tests."""

import numpy as np
import pytest

from palimpsest.evaluation.feature_drift import paired_descriptor_drift, paired_indices


def test_known_translation_reversal_and_zero_truths():
    a = np.array([[-1., 0], [-1., 0], [1., 0], [1., 0]])
    y = [0, 0, 1, 1]
    translated = paired_descriptor_drift(a, a+[2, 0], y, [1, 0])
    assert translated['before_class_gap_rms'] == pytest.approx(np.sqrt(2), abs=1e-12)
    assert translated['after_class_gap_rms'] == pytest.approx(np.sqrt(2), abs=1e-12)
    assert translated['paired_drift_rms'] == pytest.approx(np.sqrt(2), abs=1e-12)
    assert translated['mean_shift_rms'] == pytest.approx(np.sqrt(2), abs=1e-12)
    assert translated['class_gap_cosine'] == pytest.approx(1., abs=1e-12)
    flipped = paired_descriptor_drift(a, -a, y, [1, 0])
    assert flipped['class_gap_cosine'] == pytest.approx(-1., abs=1e-12)
    assert flipped['mean_shift_rms'] == 0
    assert flipped['paired_drift_rms'] == pytest.approx(np.sqrt(2), abs=1e-12)
    zero = paired_descriptor_drift(a*0, a*0, y, [0, 0])
    assert all(zero[k] == 0 for k in ('before_class_gap_rms', 'after_class_gap_rms', 'paired_drift_rms', 'mean_shift_rms'))
    assert zero['class_gap_cosine'] is None
    with pytest.raises(ValueError):
        paired_descriptor_drift(a, a, [0]*4, [1, 1])
    with pytest.raises(ValueError):
        paired_descriptor_drift(a, a*np.nan, y, [1, 1])


def test_identity_matching_and_metadata_corruptions():
    rows = [{'domain': 'rr', 'src': source, 'condition': condition,
        'variant': 'raw', 'label': label, 'scene': 'all', 'role': 'fit'}
        for source, label in (('a', 'REAL'), ('b', 'FAKE'))
        for condition in ('original', 'transfer')]
    a, b, y = paired_indices(rows, 'rr', 'original', 'raw', 'transfer', 'raw')
    np.testing.assert_array_equal(a, [0, 2]); np.testing.assert_array_equal(b, [1, 3])
    np.testing.assert_array_equal(y, [0, 1])
    for bad in (rows+rows[:1], rows[:-1], [{**rows[0], 'role': 'selection'}]+rows[1:],
                [rows[0], {**rows[1], 'label': 'FAKE'}, *rows[2:]]):
        with pytest.raises(ValueError):
            paired_indices(bad, 'rr', 'original', 'raw', 'transfer', 'raw')
