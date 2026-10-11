"""Bounded known-population controls for a higher-order mechanism hypothesis.

No image files, learned features, classifier or empirical Gaussian sampling.
"""

import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
from math import comb, factorial
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import kstat

from palimpsest.evaluation.cumulants import (
    normalized_cumulant_magnitude,
    population_cumulant4,
    unbiased_cumulant4,
)
from palimpsest.evaluation.polyphase import periodic_upsampling_matrix
from palimpsest.io.hashing import file_sha256
from palimpsest.io.provenance import load_committed_json
from palimpsest.paths import REPO_ROOT


PROTOCOL = REPO_ROOT / "configs/evaluation/cyclic_cumulant_controls.json"


def quartet_summary(columns, weights):
    explicit = columns.copy().astype(complex)
    explicit[:, (1, 3)] = explicit[:, (1, 3)].conj()
    value = population_cumulant4(explicit, weights)
    autos = [
        population_cumulant4(np.column_stack((u, u.conj(), u, u.conj())), weights)
        for u in columns.T
    ]
    return value, normalized_cumulant_magnitude(value, autos, floor=1e-12), autos


def encoded(value):
    return {"real": float(value.real), "imag": float(value.imag)}


def exact_centered_sign_outputs(kernel, latent_length):
    """Independent rational route, not the NumPy periodic matrix helper."""
    size = 2 * latent_length
    outcomes = []
    for signs in product((-1, 1), repeat=latent_length):
        row = [
            sum(
                h * signs[((i - k) % size) // 2]
                for k, h in enumerate(kernel)
                if (i - k) % 2 == 0
            )
            for i in range(size)
        ]
        mean = sum(row) / size
        outcomes.append(tuple(x - mean for x in row))
    return outcomes


def estimator_noise_variance(variances):
    """Two finite-polynomial routes; no Monte Carlo or fit parameters."""
    zero = (0, 0, 0, 0)

    def multiply(first, second):
        result = Counter()
        for a, x in first.items():
            for b, y in second.items():
                result[tuple(i + j for i, j in zip(a, b))] += x * y
        return dict(result)

    def power(poly, exponent):
        result = {zero: Fraction(1)}
        for _ in range(exponent):
            result = multiply(result, poly)
        return result

    sums = {
        degree: {
            tuple(degree if j == i else 0 for j in range(4)): Fraction(1)
            for i in range(4)
        }
        for degree in range(1, 5)
    }
    # Standard raw-sum k4 formula for n=4, independent of centered implementation.
    terms = [
        (-6, power(sums[1], 4)),
        (48, multiply(power(sums[1], 2), sums[2])),
        (-36, power(sums[2], 2)),
        (-80, multiply(sums[1], sums[3])),
        (80, sums[4]),
    ]
    polynomial = Counter()
    for coefficient, term in terms:
        for exponent, value in term.items():
            polynomial[exponent] += Fraction(coefficient, 24) * value
    squared = multiply(polynomial, polynomial)
    nodes, weights = np.polynomial.hermite_e.hermegauss(5)
    weights /= np.sqrt(2 * np.pi)
    # iid quadrature product,5 nodes x2 signs per observation;10^4 datasets.
    datasets = np.array(list(product(range(10), repeat=4)))
    rows = []
    for variance in variances:
        variance = Fraction(variance)
        moments = {}
        for order in range(9):
            value = Fraction(0)
            for noise_order in range(0, order + 1, 2):
                signal_order = order - noise_order
                if signal_order % 2:
                    continue
                gaussian = Fraction(
                    factorial(noise_order),
                    2 ** (noise_order // 2) * factorial(noise_order // 2),
                ) * variance ** (noise_order // 2)
                value += comb(order, noise_order) * gaussian
            moments[order] = value

        def expectation(poly):
            total = Fraction(0)
            for exponent, coefficient in poly.items():
                term = coefficient
                for order in exponent:
                    term *= moments[order]
                total += term
            return total

        mean = expectation(polynomial)
        exact_variance = expectation(squared) - mean**2
        support = (
            np.array([-1, 1])[:, None] + np.sqrt(float(variance)) * nodes
        ).ravel()
        probabilities = np.tile(weights / 2, 2)
        sample_values = support[datasets]
        sample_p = np.prod(probabilities[datasets], axis=1)
        centered = sample_values - sample_values.mean(axis=1, keepdims=True)
        m2 = (centered**2).mean(axis=1)
        m4 = (centered**4).mean(axis=1)
        estimates = 16 * (5 * m4 - 9 * m2**2) / 6
        quad_mean = sample_p @ estimates
        quad_variance = sample_p @ (estimates - quad_mean) ** 2
        np.testing.assert_allclose(quad_mean, float(mean), rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            quad_variance, float(exact_variance), rtol=0, atol=1e-8
        )
        assert mean == -2
        rows.append(
            {
                "noise_variance": str(variance),
                "exact_estimator_mean": str(mean),
                "exact_estimator_variance": str(exact_variance),
                "quadrature_mean": float(quad_mean),
                "quadrature_variance": float(quad_variance),
            }
        )
    assert all(
        Fraction(a["exact_estimator_variance"])
        < Fraction(b["exact_estimator_variance"])
        for a, b in zip(rows, rows[1:])
    )
    return rows


def controls(spec):
    started = perf_counter()
    tolerance = spec["gate"]["numeric_absolute_tolerance"]
    assert tolerance == 1e-10
    assert spec["finite_operator"]["quartet"] == [1, 2, 3, 6]
    caught = []

    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)

    def reject(label, actual, wrong):
        if np.allclose(actual, wrong, rtol=0, atol=tolerance):
            raise AssertionError(f"Negative control escaped: {label}")
        caught.append(label)

    signs = np.array(list(product((-1, 1), repeat=4)), dtype=float)
    output = signs @ periodic_upsampling_matrix([1, 1], 2, 4).T
    spectrum = np.fft.fft(output, axis=1, norm="ortho")
    indices = [1, 2, 3, 6]
    selected = spectrum[:, indices]
    p = np.full(16, 1 / 16)
    base, ratio, autos = quartet_summary(selected, p)
    close(base, 1j * np.sqrt(2) / 4)
    close(ratio, 1)
    # Separate coefficient-cumulant propagation, no enumerated joint moments.
    coefficients = np.fft.fft(
        periodic_upsampling_matrix([1, 1], 2, 4), axis=0, norm="ortho"
    )[indices]
    predicted = -2 * np.sum(
        coefficients[0]
        * coefficients[1].conj()
        * coefficients[2]
        * coefficients[3].conj()
    )
    close(predicted, base)

    stationary = np.array(list(product((-1, 1), repeat=8)), dtype=float)
    null_fft = np.fft.fft(stationary, axis=1, norm="ortho")
    null_p = np.full(256, 1 / 256)
    null_value, null_ratio, _ = quartet_summary(null_fft[:, indices], null_p)
    close(null_value, 0)
    close(null_ratio, 0)
    _, zero_cycle_ratio, _ = quartet_summary(null_fft[:, [1, 1, 2, 2]], null_p)
    close(zero_cycle_ratio, 1)
    reject("zero-cycle energy cumulant is a cyclic marker", zero_cycle_ratio, 0)

    gains = 0.75 + 0.25 * np.exp(-2j * np.pi * np.array(indices) / 8)
    gains *= np.exp(-2j * np.pi * np.array(indices) / 8)
    gain_value, gain_ratio, _ = quartet_summary(selected * gains, p)
    close(gain_value, base * gains[0] * gains[1].conj() * gains[2] * gains[3].conj())
    close(gain_ratio, ratio)
    # Correlated, noncircular complex noise with zero fourth cumulant.
    # This finite surrogate has Gaussian joint moments through fourth order only.
    g = np.array(list(product((-3, 0, 3), repeat=2)), dtype=float)
    gp = np.array(
        [
            float(Fraction(a) * Fraction(b))
            for a, b in product(("1/6", "2/3", "1/6"), repeat=2)
        ]
    )
    noise = g[:, :1] * np.array([[1, 2j, 1 + 1j, -2]]) + g[:, 1:] * np.array(
        [[0.5j, 1, -1j, 0.25]]
    )
    combined = (selected[:, None, :] * gains + noise[None, :, :]).reshape(-1, 4)
    combined_p = (p[:, None] * gp).ravel()
    noisy_value, noisy_ratio, _ = quartet_summary(combined, combined_p)
    close(noisy_value, gain_value)
    close(noisy_ratio, ratio)
    killed = selected.copy()
    killed[:, 0] = 0
    _, killed_ratio, _ = quartet_summary(killed, p)
    assert killed_ratio is None
    # Same improper Gaussian-moment surrogate U=V=G. Missing one pairing leaves9.
    primitive = np.array([-3, 0, 3.0])
    primitive_p = np.array([1 / 6, 2 / 3, 1 / 6])
    gaussian4 = population_cumulant4(np.column_stack([primitive] * 4), primitive_p)
    close(gaussian4, 0)
    omitted_pair = primitive_p @ primitive**4 - 2 * (primitive_p @ primitive**2) ** 2
    reject("omit pseudocovariance pairing", omitted_pair, 0)

    # Conditional Gaussian moments are integrated analytically, not sampled.
    x = np.array([-1, 1.0])
    s = 2 + x
    EY2, EY4 = np.mean(x**2 + s), np.mean(x**4 + 6 * x**2 * s + 3 * s**2)
    dependent_k4 = EY4 - 3 * EY2**2
    independent_k4 = (
        np.mean(x**4 + 6 * x**2 * 2 + 3 * 2**2) - 3 * (np.mean(x**2) + 2) ** 2
    )
    mixed_k4 = np.mean(x**4 + x**2 * s) - EY2 * np.mean(x**2) - 2 * np.mean(x**2) ** 2
    close(dependent_k4, 1)
    close(independent_k4, -2)
    close(mixed_k4, -2)
    mixed_ratio = normalized_cumulant_magnitude(
        mixed_k4, [dependent_k4, dependent_k4, -2, -2]
    )
    close(mixed_ratio, np.sqrt(2))
    reject(
        "signal-dependent noise treated as independent", dependent_k4, independent_k4
    )
    reject("normalized cumulant bounded by one", mixed_ratio, 1)

    plugin, unbiased = [], []
    for row in signs:
        columns = np.column_stack([row] * 4)
        plugin.append(population_cumulant4(columns, np.full(4, 0.25)).real)
        unbiased.append(unbiased_cumulant4(columns).real)
        close(unbiased[-1], kstat(row, n=4))
    close(np.mean(plugin), -15 / 16)
    close(np.mean(unbiased), -2)
    close(np.var(unbiased), 140 / 3)
    reject("plugin estimator treated as unbiased", np.mean(plugin), -2)
    repeated = [unbiased_cumulant4(np.full((4, 4), symbol)).real for symbol in (-1, 1)]
    close(repeated, [0, 0])
    reject("four repeated symbols treated as iid", np.mean(repeated), -2)
    # Exact Fraction distribution identity after DC removal: no finite-order test
    # can recover periodicity in this explicitly constructed transformed toy.
    kernel = tuple(Fraction(h) for h in spec["centering_counterexample"]["kernel"])
    outcomes = exact_centered_sign_outputs(kernel, 3)
    centered_shift_equal = Counter(outcomes) == Counter(
        row[-1:] + row[:-1] for row in outcomes
    )
    assert centered_shift_equal
    retained = spec["higher_order_survival"]["retained_frequencies"]
    lowpass = np.zeros_like(spectrum)
    lowpass[:, retained] = spectrum[:, retained]
    spatial = np.fft.ifft(lowpass, axis=1, norm="ortho").real
    covariance = spatial.T @ spatial / len(spatial)
    second_shift_residual = float(
        np.max(np.abs(covariance - np.roll(covariance, (1, 1), axis=(0, 1))))
    )
    close(second_shift_residual, 0)
    _, higher_ratio, _ = quartet_summary(
        lowpass[:, spec["higher_order_survival"]["quartet"]], p
    )
    close(higher_ratio, 1)
    noise_variances = estimator_noise_variance(
        spec["noise_estimator_variance"]["independent_gaussian_variances"]
    )
    shifted_populations, per_record_ratios = [], []
    for shift in spec["phase_grouping"]["per_record_global_shifts"]:
        population = np.fft.fft(np.roll(output, shift, axis=1), axis=1, norm="ortho")
        _, per_record_ratio, _ = quartet_summary(population[:, indices], p)
        close(per_record_ratio, 1)
        per_record_ratios.append(per_record_ratio)
        shifted_populations.append(population[:, indices])
    pooled = np.concatenate(shifted_populations)
    pooled_value, pooled_ratio, _ = quartet_summary(
        pooled, np.full(len(pooled), 1 / len(pooled))
    )
    close(pooled_value, 0)
    close(pooled_ratio, 0)
    reject("unaligned complex pooling preserves each record's marker", pooled_ratio, 1)
    return {
        "scope": spec["scope"],
        "elapsed_seconds": perf_counter() - started,
        "population": {
            "base_quartet": encoded(base),
            "base_ratio": ratio,
            "stationary_null_ratio": null_ratio,
            "stationary_zero_cycle_ratio": zero_cycle_ratio,
            "gain_ratio": gain_ratio,
            "noise_ratio": noisy_ratio,
            "killed_ratio": killed_ratio,
            "omitted_pair_spurious": omitted_pair,
            "coefficient_route_error": float(abs(predicted - base)),
        },
        "noise_counterexample": {
            "independent_k4": independent_k4,
            "dependent_k4": dependent_k4,
            "mixed_k4": mixed_k4,
            "mixed_ratio": mixed_ratio,
        },
        "estimation": {
            "sample_size": 4,
            "enumerated_datasets": 16,
            "plugin_mean": float(np.mean(plugin)),
            "unbiased_mean": float(np.mean(unbiased)),
            "unbiased_variance": float(np.var(unbiased)),
            "repeated_symbol_mean": float(np.mean(repeated)),
        },
        "centering_counterexample": {
            "states": len(outcomes),
            "entire_distribution_shift_equal": centered_shift_equal,
        },
        "higher_order_survival": {
            "retained_frequencies": retained,
            "second_order_shift_residual": second_shift_residual,
            "fourth_normalized_magnitude": higher_ratio,
        },
        "noise_estimator_variance": noise_variances,
        "phase_grouping": {
            "per_record_ratios": per_record_ratios,
            "unaligned_pooled_quartet": encoded(pooled_value),
            "unaligned_pooled_ratio": pooled_ratio,
        },
        "negative_controls_caught": caught,
        "excluded_claims": spec["excluded_claims"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec, digest = load_committed_json(PROTOCOL, args.protocol_revision, REPO_ROOT)
    result = controls(spec)
    result.update(
        protocol_revision=args.protocol_revision,
        protocol_sha256=digest,
        code_sha256=file_sha256(Path(__file__)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": result["elapsed_seconds"],
                "negative_controls": len(result["negative_controls_caught"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
