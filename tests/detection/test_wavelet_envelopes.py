"""Texture-phase, translation and useless-stability controls before real data."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.wavelet_envelopes import (
    FIRST_NAMES, SECOND_NAMES, FEATURE_NAMES, envelope_statistics, extract_envelopes, morlet_filters,
)


def test_constant_and_even_pixel_periodic_translation():
    assert len(FIRST_NAMES) == 24 and len(SECOND_NAMES) == 96 and len(FEATURE_NAMES) == 120
    np.testing.assert_array_equal(envelope_statistics(np.full((64, 64), 128.)), np.zeros(120))
    rng = np.random.default_rng(902)
    gray = rng.normal(128, 20, (64, 64))
    np.testing.assert_allclose(envelope_statistics(gray), envelope_statistics(np.roll(gray, (2, 4), (0, 1))),
                               atol=1e-13, rtol=0)
    np.testing.assert_allclose(envelope_statistics(gray), envelope_statistics(gray+30), atol=1e-13, rtol=0)
    for kernel in morlet_filters(gray.shape):
        assert kernel[0, 0] == 0 and not kernel.flags.writeable
    # Zero pair distance on constant images cannot be evidence of class separation.
    np.testing.assert_array_equal(envelope_statistics(np.full((64, 64), 10.)),
                                  envelope_statistics(np.full((64, 64), 200.)))


def test_same_power_different_spatial_phase_and_validation():
    # Real impulses and a phase-randomized version have exactly equal Fourier
    # magnitudes. Preserve Hermitian symmetry by taking phases of real noise.
    gray = np.zeros((64, 64)); gray[16, 16] = 100; gray[40, 36] = 80
    rng = np.random.default_rng(775)
    spectrum = np.fft.fft2(gray)
    phase = np.angle(np.fft.fft2(rng.normal(size=gray.shape)))
    shuffled = np.fft.ifft2(np.abs(spectrum)*np.exp(1j*phase)).real
    np.testing.assert_allclose(np.abs(np.fft.fft2(shuffled)), np.abs(spectrum), atol=2e-13, rtol=1e-13)
    a, b = envelope_statistics(gray), envelope_statistics(shuffled)
    assert np.linalg.norm(a[24:]-b[24:]) > .1
    assert np.isfinite(extract_envelopes(np.full((1, 1, 3), 100, np.uint8)).values).all()
    for bad in (np.zeros((31, 32)), np.full((32, 32), np.nan), np.zeros(32)):
        with pytest.raises(ValueError): envelope_statistics(bad)
    with pytest.raises(ValueError): extract_envelopes(np.zeros((64, 64, 3), float))
