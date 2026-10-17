"""Noise-invariant cross-frequency covariance does not imply invariant coherence."""

import numpy as np
import pytest

from fractions import Fraction

from experiments.origin_detection.non_neural.generator_mechanisms.audit_spectral_transport import (
    _average_population,
    _check_estimate,
    controls,
    estimation_controls,
)
from palimpsest.evaluation.polyphase import (
    frequency_covariance,
    spectral_coherence_magnitude,
)


def test_registered_filter_shift_and_noise_controls():
    r = controls()
    assert r["negative_controls_caught"] == 5
    assert r["base_C26"]["imag"] == pytest.approx(-1)
    assert r["white_C26"]["imag"] == pytest.approx(-1)
    assert r["white_coherence"] == pytest.approx(0.5)
    assert r["lowpass_max_off_diagonal"] < 1e-12
    assert r["lowpass_mean_square"] > 0.6


@pytest.mark.parametrize(
    "bad",
    (
        [],
        np.empty((0, 0)),
        np.ones((2, 3)),
        [[np.nan]],
        [[1, 2], [0, 1]],
        [[1, 2], [2, 1]],
    ),
)
def test_invalid_frequency_covariance_refused(bad):
    with pytest.raises(ValueError):
        frequency_covariance(bad)


def test_zero_power_is_undefined_not_classification():
    assert spectral_coherence_magnitude(np.zeros((4, 4)), 1, 2) is None


@pytest.mark.parametrize("first,second", ((True, 1), (0, 4), (-1, 0), (0, 1.5)))
def test_invalid_frequency_indices_refused(first, second):
    with pytest.raises(ValueError):
        spectral_coherence_magnitude(np.eye(4), first, second)


def test_exact_small_sample_and_shared_window_controls():
    r = estimation_controls(max_windows=2)
    assert r["positive_fixtures"] == r["wrong_mse_caught"] == 4
    cases = {(c["model"], c["windows"]): c for c in r["fixtures"]}
    assert cases[("coupled_repeat", 2)]["mse"] == "3/4"
    assert cases[("coupled_repeat", 2)]["shared_window_mse"] == "3/2"
    assert cases[("stationary_null", 2)]["mse"] == "1/4"
    assert cases[("stationary_null", 2)]["nonzero_probability"] == "11/16"
    with pytest.raises(ValueError):
        _check_estimate(
            (Fraction(0), Fraction(0)),
            Fraction(1, 4),
            (Fraction(0), Fraction(0)),
            Fraction(0),
        )


@pytest.mark.parametrize(
    "population,windows",
    (({}, 1), ({(0, 0): Fraction(1, 2)}, 1), ({(0, 0): Fraction(1)}, 3)),
)
def test_invalid_exact_average_input_refused(population, windows):
    with pytest.raises(ValueError):
        _average_population(population, windows)
