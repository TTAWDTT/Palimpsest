"""Pinned operator assay refuses changed source before optional Torch imports."""

import pytest
import numpy as np

from experiments.origin_detection.non_neural.generator_mechanisms.audit_ldm_operator import (
    _checked_weights,
    _pinned_class,
    assay,
    trained_footprint,
)


def test_changed_source_refused_before_execution(tmp_path):
    source = tmp_path / "changed.py"
    source.write_text('raise RuntimeError("must never execute")\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Source SHA"):
        _pinned_class(source)


def test_unregistered_dimensions_refused_before_source_or_torch(tmp_path):
    with pytest.raises(ValueError, match="fixture length"):
        assay(tmp_path / "missing.py", 100)


def test_trained_footprint_indexing_known_box_filter():
    weight = np.zeros((1, 1, 3, 3))
    weight[0, 0, 1:, 1:] = 0.25
    even = trained_footprint(weight, 0, 0, 1)
    odd = trained_footprint(weight, 1, 1, 1)
    expected_even = np.zeros((1, 1, 4, 4))
    expected_even[0, 0, 1, 1] = 1
    expected_odd = np.zeros((1, 1, 4, 4))
    expected_odd[0, 0, 1:3, 1:3] = 0.25
    np.testing.assert_array_equal(even, expected_even)
    np.testing.assert_array_equal(odd, expected_odd)


def test_changed_weights_refused_before_loading(tmp_path):
    p = tmp_path / "changed.safetensors"
    p.write_bytes(b"not weights")
    with pytest.raises(ValueError, match="byte length"):
        _checked_weights(p)


@pytest.mark.parametrize(
    "weight,r,c",
    (
        (np.ones((1, 1, 2, 2)), 0, 0),
        (np.ones((1, 1, 3, 3)), 2, 0),
        (np.full((1, 1, 3, 3), np.nan), 0, 0),
    ),
)
def test_invalid_trained_footprint_refused(weight, r, c):
    with pytest.raises(ValueError):
        trained_footprint(weight, r, c, 1)
