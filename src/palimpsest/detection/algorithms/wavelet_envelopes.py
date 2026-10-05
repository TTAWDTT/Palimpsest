"""Fixed finite Morlet-envelope statistics, without neural encoders.

This discretized texture descriptor is inspired by scattering cascades. Its
truncation, contrast normalization, resizing and borders do not inherit a
continuous scattering deformation theorem or guarantee forensic discrimination.
"""

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import cv2
import numpy as np

from palimpsest.contracts import Prediction
from .paired_stability import StableRule
from .residual_statistics.features import FEATURE_NAMES as RESIDUAL_NAMES, extract_features

CENTERS = (.25, .125, .0625)
ANGLES = tuple(np.arange(4)*np.pi/4)
FIRST_NAMES = tuple(f'envelope1/{j}/{o}/{m}' for j in range(3) for o in range(4) for m in ('mean', 'rms'))
SECOND_NAMES = tuple(f'envelope2/{j1}/{o1}/{j2}/{o2}/{m}'
                     for j1 in range(3) for o1 in range(4) for j2 in range(j1+1, 3)
                     for o2 in range(4) for m in ('mean', 'rms'))
FEATURE_NAMES = FIRST_NAMES + SECOND_NAMES


@lru_cache(maxsize=32)
def morlet_filters(shape, step=1.):
    """Periodic frequency-grid filters with a zero DC correction, peak scaled."""
    h, w = shape
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij')
    filters = []
    for center in CENTERS:
        k = center*step
        sigma = k/2
        for angle in ANGLES:
            shifted = np.exp(-((fx-k*np.cos(angle))**2+(fy-k*np.sin(angle))**2)/(2*sigma**2))
            dc = np.exp(-(fx**2+fy**2)/(2*sigma**2))*np.exp(-k*k/(2*sigma**2))
            kernel = shifted-dc
            kernel[0, 0] = 0.
            kernel /= np.max(np.abs(kernel))
            kernel.setflags(write=False)
            filters.append(kernel)
    return tuple(filters)


def envelope_statistics(gray):
    """24 first-order and96 second-order mean/RMS coefficients on a gray grid.

    The envelope is area-downsampled by2 before a coarser wavelet. Second-order
    moments are divided by parent mean+1e-3. These are engineering choices,
    not a certified tight frame or the original paper's complete scattering.
    """
    gray = np.asarray(gray, float)
    if gray.ndim != 2 or min(gray.shape) < 32 or not np.isfinite(gray).all():
        raise ValueError('Invalid envelope gray grid')
    z = gray-gray.mean()
    z /= np.sqrt(np.mean(z*z))+1.
    spectrum = np.fft.fft2(z)
    first, second = [], []
    for index, kernel in enumerate(morlet_filters(gray.shape)):
        envelope = np.abs(np.fft.ifft2(spectrum*kernel))
        mean = float(envelope.mean())
        first.extend((mean, float(np.sqrt(np.mean(envelope*envelope)))))
        j1 = index//4
        if j1 == 2: continue
        h, w = gray.shape
        coarse = cv2.resize(envelope, (w//2, h//2), interpolation=cv2.INTER_AREA)
        transform = np.fft.fft2(coarse)
        bank = morlet_filters(coarse.shape, 2.)
        for j2 in range(j1+1, 3):
            for o2 in range(4):
                response = np.abs(np.fft.ifft2(transform*bank[4*j2+o2]))
                second.extend((float(response.mean())/(mean+1e-3),
                               float(np.sqrt(np.mean(response*response)))/(mean+1e-3)))
    values = np.array([*first, *second])
    if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError('Envelope feature schema/numerics differ')
    return values


@dataclass(frozen=True)
class EnvelopeFeatures:
    values: np.ndarray
    preprocess_ms: float
    statistics_ms: float


def extract_envelopes(image):
    start = perf_counter()
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
        raise ValueError('Expected nonempty RGB uint8 image')
    h, w = rgb.shape[:2]
    factor = min(1., 128/max(h, w))
    if factor < 1:
        rgb = cv2.resize(rgb, (max(1, round(w*factor)), max(1, round(h*factor))), interpolation=cv2.INTER_AREA)
    gray = (.299*rgb[..., 0]+.587*rgb[..., 1]+.114*rgb[..., 2]).astype(float)
    h, w = gray.shape
    if min(h, w) < 32:
        gray = np.pad(gray, ((0, max(0, 32-h)), (0, max(0, 32-w))), mode='reflect' if min(h, w)>1 else 'edge')
    ready = perf_counter()
    values = envelope_statistics(gray)
    return EnvelopeFeatures(values, 1000*(ready-start), 1000*(perf_counter()-ready))


class EnvelopeDetector:
    name = 'wavelet-envelope-score'

    def __init__(self, rule: StableRule):
        allowed = (FIRST_NAMES, SECOND_NAMES, FEATURE_NAMES, RESIDUAL_NAMES+FEATURE_NAMES)
        if rule.feature_names not in allowed:
            raise ValueError('Envelope readout schema differs')
        self.rule = rule
        self.rule_sha256 = rule.fingerprint

    def predict(self, image):
        start = perf_counter()
        f = extract_envelopes(image)
        names = self.rule.feature_names
        values = f.values if names == FEATURE_NAMES else f.values[:24] if names == FIRST_NAMES else f.values[24:]
        if names == RESIDUAL_NAMES+FEATURE_NAMES:
            values = np.r_[extract_features(image).values, f.values]
        score = float(self.rule.score(values[None])[0])
        return Prediction(self.name, score, self.rule.threshold, 'statistical_score',
                          timing_ms={'preprocess_ms': f.preprocess_ms, 'statistics_ms': f.statistics_ms,
                                     'predict_ms': 1000*(perf_counter()-start)},
                          metadata={'rule_sha256': self.rule_sha256,
                                    'fit_manifest_sha256': self.rule.fit_manifest_sha256})
