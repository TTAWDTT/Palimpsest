"""One-shot finite spectral transport controls, no image classification.

Use covariance from known periodic operators and separately transform the
postprocessing operator to check filter, shift and noise assumptions.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.evaluation.polyphase import (
    frequency_covariance,
    periodic_upsampling_matrix,
    spectral_coherence_magnitude,
)


def _check(actual, expected, name):
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(f"Known spectral answer mismatch: {name}")


def controls():
    started = perf_counter()
    a = periodic_upsampling_matrix([1, 1], 2, 4)
    covariance = a @ a.T
    size = len(covariance)
    spectrum = frequency_covariance(covariance)
    _check(spectrum[2, 6], -1j, "base C26")
    shifted = np.roll(covariance, (1, 1), axis=(0, 1))
    shift_spectrum = frequency_covariance(shifted)
    _check(shift_spectrum[2, 6], 1j, "shifted C26")
    _check(np.abs(shift_spectrum), np.abs(spectrum), "shifted full magnitudes")
    b = periodic_upsampling_matrix([0.5, 0.5], 1, size)
    filtered = b @ covariance @ b.T
    filtered_spectrum = frequency_covariance(filtered)
    transfer = np.fft.fft([0.5, 0.5] + [0.0] * (size - 2))
    _check(
        filtered_spectrum,
        transfer[:, None] * spectrum * transfer.conj()[None, :],
        "filter law",
    )
    _check(filtered_spectrum[2, 6], -0.5, "filtered C26")
    white = covariance + np.eye(size)
    white_spectrum = frequency_covariance(white)
    _check(white_spectrum, spectrum + np.eye(size), "independent white noise")
    _check(spectral_coherence_magnitude(covariance, 2, 6), 1, "base coherence")
    _check(spectral_coherence_magnitude(white, 2, 6), 0.5, "white noise coherence")
    nonstationary = covariance + np.diag([2.0, 0.0] * 4)
    nonstationary_spectrum = frequency_covariance(nonstationary)
    _check(nonstationary_spectrum[2, 6], 1 - 1j, "nonstationary noise C26")
    # Fixed lowpass with frequencies0,1,7; its real circular impulse response.
    mask = np.zeros(size)
    mask[[0, 1, 7]] = 1
    kernel = np.fft.ifft(mask).real
    lowpass = periodic_upsampling_matrix(kernel, 1, size)
    low_covariance = lowpass @ covariance @ lowpass.T
    low_spectrum = frequency_covariance(low_covariance)
    off_diagonal = low_spectrum - np.diag(np.diag(low_spectrum))
    _check(off_diagonal, 0, "lowpass erases off-diagonal covariance")
    expected_energy = (4 + np.sqrt(2)) / 8
    _check(
        np.trace(low_covariance) / size,
        expected_energy,
        "nonzero retained lowpass energy",
    )
    balanced = periodic_upsampling_matrix([1, 2 / 3, 0, -1 / 3, 0, 2 / 3], 2, 3)
    balanced_spectrum = frequency_covariance(balanced @ balanced.T)
    _check(
        balanced_spectrum - np.diag(np.diag(balanced_spectrum)),
        0,
        "balanced covariance",
    )
    # Each false statement is explicitly checked and must throw.
    wrongs = (
        ("shift sign", shift_spectrum[2, 6], -1j),
        (
            "missing conjugation",
            filtered_spectrum,
            transfer[:, None] * spectrum * transfer[None, :],
        ),
        ("noise invariant coherence", spectral_coherence_magnitude(white, 2, 6), 1),
        ("nonstationary noise immunity", nonstationary_spectrum[2, 6], -1j),
        ("lowpass keeps C26", low_spectrum[2, 6], -1j),
    )
    caught = []
    for name, actual, wrong in wrongs:
        try:
            _check(actual, wrong, name)
        except ValueError:
            caught.append(name)
        else:
            raise ValueError(f"Wrong spectral statement survived: {name}")
    value = lambda z: {
        "real": float(np.real(z)),
        "imag": float(np.imag(z)),
        "magnitude": float(abs(z)),
    }
    return {
        "base_C26": value(spectrum[2, 6]),
        "shift_C26": value(shift_spectrum[2, 6]),
        "filter_C26": value(filtered_spectrum[2, 6]),
        "white_C26": value(white_spectrum[2, 6]),
        "nonstationary_noise_C26": value(nonstationary_spectrum[2, 6]),
        "base_coherence": spectral_coherence_magnitude(covariance, 2, 6),
        "white_coherence": spectral_coherence_magnitude(white, 2, 6),
        "lowpass_max_off_diagonal": float(np.max(abs(off_diagonal))),
        "lowpass_mean_square": float(np.trace(low_covariance) / size),
        "balanced_max_off_diagonal": float(
            np.max(abs(balanced_spectrum - np.diag(np.diag(balanced_spectrum))))
        ),
        "negative_controls_caught": len(caught),
        "corruptions": caught,
        "elapsed_seconds": perf_counter() - started,
        "scope": "Known finite centered population covariance; no single-image estimator, AI specificity, JPEG or recapture validation",
    }


def _window_population(model):
    """Exact frequency-product population via directly written4-point DFT."""
    population = defaultdict(Fraction)
    for z in product((-1, 1), repeat=4):
        nyquist = Fraction(z[0] - z[1] + z[2] - z[3], 2)
        if model == "coupled_repeat":
            value = (Fraction(0), -(nyquist**2))
        elif model == "stationary_null":
            real = Fraction(z[0] - z[2], 2)
            imag = Fraction(-z[1] + z[3], 2)
            value = (real * nyquist, imag * nyquist)
        else:
            raise ValueError("Unknown registered observation model")
        population[value] += Fraction(1, 16)
    return dict(population)


def _average_population(population, windows):
    """Rational probability convolution; no Monte Carlo observations."""
    if (
        not isinstance(windows, int)
        or isinstance(windows, bool)
        or windows not in (1, 2, 4, 8, 16)
    ):
        raise ValueError("Window count is not registered")
    if not population or sum(population.values()) != 1 or min(population.values()) < 0:
        raise ValueError("One-window probability mass must be nonnegative and sum to1")
    distribution = {(Fraction(0), Fraction(0)): Fraction(1)}
    for _ in range(windows):
        next_distribution = defaultdict(Fraction)
        for (a, b), weight in distribution.items():
            for (x, y), probability in population.items():
                next_distribution[(a + x, b + y)] += weight * probability
        distribution = next_distribution
    return {
        (a / windows, b / windows): weight for (a, b), weight in distribution.items()
    }


def _distribution_moments(distribution, target):
    mean = tuple(
        sum(value[i] * weight for value, weight in distribution.items()) for i in (0, 1)
    )
    mse = sum(
        ((value[0] - target[0]) ** 2 + (value[1] - target[1]) ** 2) * weight
        for value, weight in distribution.items()
    )
    nonzero = sum(weight for value, weight in distribution.items() if value != (0, 0))
    return mean, mse, nonzero


def _shared_average_population(population, windows):
    averaged = {}
    for value, weight in population.items():
        copies = [value] * windows
        result = tuple(sum(copy[i] for copy in copies) / windows for i in (0, 1))
        averaged[result] = weight
    return averaged


def _check_estimate(actual_mean, actual_mse, expected_mean, expected_mse):
    if actual_mean != expected_mean or actual_mse != expected_mse:
        raise ValueError("Known mean-estimator distribution mismatch")


def estimation_controls(max_windows=16):
    """Known means and mean-square errors; independence is an explicit assumption."""
    if max_windows not in (2, 16):
        raise ValueError("Registered pilot/full max_windows must be2 or16")
    started = perf_counter()
    results = []
    corruptions = []
    models = (
        ("coupled_repeat", (Fraction(0), Fraction(-1)), Fraction(3, 2)),
        ("stationary_null", (Fraction(0), Fraction(0)), Fraction(1, 2)),
    )
    for model, target, single_mse in models:
        population = _window_population(model)
        for windows in (1, 2, 4, 8, 16):
            if windows > max_windows:
                continue
            distribution = _average_population(population, windows)
            mean, mse, nonzero = _distribution_moments(distribution, target)
            _check_estimate(mean, mse, target, single_mse / windows)
            shared = _shared_average_population(population, windows)
            shared_mean, shared_mse, _ = _distribution_moments(shared, target)
            _check_estimate(shared_mean, shared_mse, target, single_mse)
            if model == "stationary_null" and windows in (1, 2):
                expected_nonzero = Fraction(1, 2) if windows == 1 else Fraction(11, 16)
                if nonzero != expected_nonzero:
                    raise ValueError("Null finite-sample nonzero probability changed")
            try:
                _check_estimate(mean, mse, target, mse + 1)
            except ValueError:
                corruptions.append(f"{model}/M{windows}/wrong_mse")
            else:
                raise ValueError("Wrong MSE was not caught")
            results.append(
                {
                    "model": model,
                    "windows": windows,
                    "mean": [str(v) for v in mean],
                    "mse": str(mse),
                    "shared_window_mse": str(shared_mse),
                    "nonzero_probability": str(nonzero),
                    "probability_states": len(distribution),
                }
            )
    if max_windows == 16 and len(results) != 10:
        raise ValueError("Full registered estimate denominator changed")
    # Test the false iid claim on the actual fully shared-window computation.
    source_population = _window_population("coupled_repeat")
    shared = _shared_average_population(source_population, 2)
    mean, mse, _ = _distribution_moments(shared, (Fraction(0), Fraction(-1)))
    try:
        _check_estimate(mean, mse, (Fraction(0), Fraction(-1)), Fraction(3, 4))
    except ValueError:
        shared_error_caught = True
    else:
        raise ValueError("False independence assumption survived")
    return {
        "fixtures": results,
        "positive_fixtures": len(results),
        "wrong_mse_caught": len(corruptions),
        "corruptions": corruptions,
        "false_independence_caught": shared_error_caught,
        "elapsed_seconds": perf_counter() - started,
        "scope": "Exact toy frequency-product distributions and independent vs perfectly shared windows; not image-patch independence, a significance threshold, AI detection or robustness validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=("transport", "estimation"), default="transport"
    )
    parser.add_argument("--max-windows", type=int, choices=(2, 16), default=16)
    args = parser.parse_args()
    result = (
        controls()
        if args.suite == "transport"
        else estimation_controls(args.max_windows)
    )
    result["code_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    import palimpsest.evaluation.polyphase as module

    result["operator_code_sha256"] = sha256(
        Path(module.__file__).read_bytes()
    ).hexdigest()
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
