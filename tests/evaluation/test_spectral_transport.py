"""Noise-invariant cross-frequency covariance does not imply invariant coherence."""

import numpy as np
import pytest

from experiments.origin_detection.non_neural.generator_mechanisms.audit_spectral_transport import controls
from palimpsest.evaluation.polyphase import frequency_covariance, spectral_coherence_magnitude


def test_registered_filter_shift_and_noise_controls():
    r=controls()
    assert r['negative_controls_caught']==5
    assert r['base_C26']['imag']==pytest.approx(-1)
    assert r['white_C26']['imag']==pytest.approx(-1)
    assert r['white_coherence']==pytest.approx(0.5)
    assert r['lowpass_max_off_diagonal']<1e-12
    assert r['lowpass_mean_square']>0.6


@pytest.mark.parametrize('bad', ([],np.empty((0,0)),np.ones((2,3)),[[np.nan]],[[1,2],[0,1]],[[1,2],[2,1]]))
def test_invalid_frequency_covariance_refused(bad):
    with pytest.raises(ValueError): frequency_covariance(bad)


def test_zero_power_is_undefined_not_classification():
    assert spectral_coherence_magnitude(np.zeros((4,4)),1,2) is None


@pytest.mark.parametrize('first,second', ((True,1),(0,4),(-1,0),(0,1.5)))
def test_invalid_frequency_indices_refused(first,second):
    with pytest.raises(ValueError): spectral_coherence_magnitude(np.eye(4),first,second)
