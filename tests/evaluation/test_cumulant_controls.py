"""Joint-moment conventions and iid-estimator boundaries."""

from itertools import product

import numpy as np
import pytest
from scipy.stats import kstat

from palimpsest.evaluation.cumulants import (
    normalized_cumulant_magnitude,
    population_cumulant4,
    unbiased_cumulant4,
    zero_mean_group_decomposition,
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


@pytest.mark.parametrize("complex_columns", (False, True))
def test_zero_conditional_cumulants_do_not_imply_zero_mixture(complex_columns):
    z = np.array([-np.sqrt(3), 0, np.sqrt(3)])
    x = np.concatenate((z, 2 * z))
    factors = np.array([1, 1j, 2, -1j]) if complex_columns else np.ones(4)
    row = zero_mean_group_decomposition(
        x[:, None] * factors, np.tile([1 / 12, 1 / 3, 1 / 12], 2), [0, 0, 0, 1, 1, 1]
    )
    scale = 2 if complex_columns else 1
    assert row["weighted_conditional"] == pytest.approx(0, abs=1e-10)
    assert row["joint"] == pytest.approx(scale * 27 / 4)
    assert row["covariance_mixing"] == pytest.approx(scale * 27 / 4)
    assert row["residual"] == pytest.approx(0, abs=1e-10)


def test_sign_cumulant_survives_variance_mixture_decomposition():
    x = np.array([-1, 1, -2, 2])
    row = zero_mean_group_decomposition(
        np.repeat(x[:, None], 4, axis=1), [0.25] * 4, [0, 0, 1, 1]
    )
    assert row["weighted_conditional"] == pytest.approx(-17)
    assert row["joint"] == pytest.approx(-41 / 4)
    assert row["covariance_mixing"] == pytest.approx(27 / 4)


@pytest.mark.parametrize("weights", ([0.25] * 4, [0.5, 0.5, 0, 0]))
def test_unknown_conditional_means_and_empty_mass_are_refused(weights):
    x = np.array([-1, 1, 0, 2])
    with pytest.raises(ValueError):
        zero_mean_group_decomposition(
            np.repeat(x[:, None], 4, axis=1), weights, [0, 0, 1, 1]
        )
