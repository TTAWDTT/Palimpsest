"""Population-free spatial summaries and measured-input prediction controls."""

import numpy as np
import pytest

from experiments.origin_detection.non_neural.generator_mechanisms.audit_decoder_chain import (
    committed_protocol,
    indexed_prediction,
)
from palimpsest.evaluation.polyphase import spatial_phase_moments


def test_single_field_phase_moments_known_values():
    field = np.array(
        [[0, 0, 1, 2], [0, 0, 3, 4], [2, 4, 3, 6], [6, 8, 9, 12]], dtype=float
    )
    r = spatial_phase_moments(field)
    assert r["means"] == [[1.5, 3], [4.5, 6]]
    assert r["variances"] == [[1.25, 5], [11.25, 20]]
    assert r["counts"] == [[4, 4], [4, 4]]
    assert spatial_phase_moments(np.ones((4, 4)))["variance_cv"] is None


def test_measured_input_known_unit_kernel():
    hidden = np.arange(16, dtype=float).reshape(1, 4, 4)
    weight = np.zeros((1, 1, 3, 3))
    weight[0, 0, 1, 1] = 1
    for r in (0, 1):
        for c in (0, 1):
            actual = indexed_prediction(hidden, weight, np.array([2.0]), r, c, 1)
            np.testing.assert_array_equal(actual, hidden[:, 1:3, 1:3] + 2)


def test_measured_input_known_forward_mean():
    hidden = np.arange(16, dtype=float).reshape(1, 4, 4)
    weight = np.zeros((1, 1, 3, 3))
    weight[0, 0, 1:, 1:] = 0.25
    actual = indexed_prediction(hidden, weight, np.array([0.0]), 1, 1, 1)
    np.testing.assert_array_equal(actual, np.array([[[7.5, 8.5], [11.5, 12.5]]]))


@pytest.mark.parametrize("bad", ([], np.zeros((3, 4)), np.full((4, 4), np.nan)))
def test_bad_spatial_field_refused(bad):
    with pytest.raises(ValueError):
        spatial_phase_moments(bad)


def test_measured_input_bilinear_coordinate_weights():
    hidden = np.arange(16, dtype=float).reshape(1, 4, 4)
    weight = np.zeros((1, 1, 3, 3))
    weight[0, 0, 1, 1] = 1
    expected = {
        (0, 0): np.array([[[3.75, 4.75], [7.75, 8.75]]]),
        (1, 1): np.array([[[6.25, 7.25], [10.25, 11.25]]]),
    }
    for (r, c), values in expected.items():
        np.testing.assert_array_equal(
            indexed_prediction(hidden, weight, np.zeros(1), r, c, 1, mode="bilinear"),
            values,
        )


def test_protocol_revision_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="full commit"):
        committed_protocol(tmp_path / "missing.json", "HEAD")
