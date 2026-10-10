"""Known-answer order controls and identical-marginal joint dependence."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.color_relations import (
    FEATURE_NAMES, ORBITS, cross_counts, extract_colors, order_histograms, product_counts,
)


def test_order_definition_ties_and_monotone_boundary():
    image = np.full((3, 3, 3), 2., dtype=float)
    image[0, 0] = [1, 2, 2]  # Strict product dominance includes component ties.
    image[0, 1] = [3, 1, 2]  # Incomparable.
    image[0, 2] = [3, 2, 2]  # Strictly greater.
    result = product_counts(image)
    assert [x.item() for x in result] == [1, 1, 5]
    order, ties = order_histograms(image)
    assert order[10] == 1 and ties[5] == 1
    transformed = image**np.array([1., 2., 3.])
    assert all(np.array_equal(a, b) for a, b in zip(result, product_counts(transformed)))
    matrix = np.array([[1, 1, .1], [1, 2, .2], [1, 3, .4]])
    a, b = np.array([2., 0., 0.]), np.array([0., 2., 0.])
    assert not (np.all(a <= b) or np.all(b <= a))
    assert np.all(matrix@a <= matrix@b) and np.any(matrix@a != matrix@b)


def test_joint_control_preserves_marginals_changes_dependence():
    a = np.array([[[2, 2, 2], [-2, -2, -2]]], dtype=np.int8)
    b = np.array([[[2, -2, 2], [-2, 2, -2]]], dtype=np.int8)
    assert all(np.array_equal(np.sort(a[..., c].ravel()), np.sort(b[..., c].ravel())) for c in range(3))
    assert not np.array_equal(cross_counts(a), cross_counts(b))
    assert np.array_equal(cross_counts(a), cross_counts(-a))
    assert len(ORBITS) == 63 and cross_counts(a).sum() == 2
    for image in (np.full((1, 1, 3), 128, np.uint8),
                  np.random.default_rng(4).integers(0, 256, (59, 73, 3), dtype=np.uint8)):
        value = extract_colors(image).values
        assert len(value) == len(FEATURE_NAMES) == 351
        assert np.isfinite(value).all() and np.all((value >= 0) & (value <= 1))
        assert np.allclose([value[s:e].sum() for s, e in
                            ((0,45),(45,90),(90,135),(135,144),(144,153),(153,162),(162,225),(225,288),(288,351))], 1)
    with pytest.raises(ValueError):
        extract_colors(np.zeros((2, 2, 3), float))
    with pytest.raises(ValueError):
        cross_counts(np.zeros((2, 2, 3), float))
