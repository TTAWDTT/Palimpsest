"""Joint-moment conventions and iid-estimator boundaries."""

from itertools import product

import numpy as np
import pytest
from scipy.stats import kstat

from palimpsest.evaluation.cumulants import (
    normalized_cumulant_magnitude,
    population_cumulant4,
    unbiased_cumulant4,
)


def test_improper_gaussian_moments_need_all_three_pairings():
    g = np.array([-3, 0, 3])
    assert population_cumulant4(np.column_stack([g] * 4), [1 / 6, 2 / 3, 1 / 6]) == 0


def test_complex_conjugation_and_translation_are_explicit():
    x = np.array([-1, 1], dtype=complex) * (1 + 2j)
    columns = np.column_stack((x, x.conj(), x, x.conj()))
    np.testing.assert_allclose(population_cumulant4(columns + 7j, [0.5, 0.5]), -50)


def test_iid_kstatistic_matches_scalar_reference_and_has_large_variance():
    results = []
    for row in product((-1, 1), repeat=4):
        value = unbiased_cumulant4(np.column_stack([row] * 4)).real
        np.testing.assert_allclose(value, kstat(row, n=4), rtol=0, atol=1e-12)
        results.append(value)
    np.testing.assert_allclose(np.mean(results), -2)
    np.testing.assert_allclose(np.var(results), 140 / 3)


def test_normalization_is_not_bounded_and_degeneracy_is_explicit():
    assert normalized_cumulant_magnitude(-2, [1, 1, -2, -2]) == pytest.approx(
        np.sqrt(2)
    )
    assert normalized_cumulant_magnitude(0, [0, 1, 1, 1]) is None
    assert normalized_cumulant_magnitude(1, [0.01, 1, 1, 1], floor=0.1) is None


@pytest.mark.parametrize("bad", ([], np.ones((4, 3)), np.full((4, 4), np.nan)))
def test_bad_joint_values_refused(bad):
    with pytest.raises(ValueError):
        population_cumulant4(bad, [0.25] * 4)


@pytest.mark.parametrize("bad", ([0.5, 0.6], [-0.1, 1.1], [np.nan, 1]))
def test_bad_population_probabilities_refused(bad):
    with pytest.raises(ValueError):
        population_cumulant4(np.zeros((2, 4)), bad)


def test_too_few_vectors_and_invalid_floor_refused():
    with pytest.raises(ValueError):
        unbiased_cumulant4(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        normalized_cumulant_magnitude(1, [1, 1, 1, 1], floor=-1)
