"""Fixed random Fourier kernel features over fitted pixel-statistic scales."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule


@dataclass(frozen=True)
class FourierMap:
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    frequencies: tuple[tuple[float, ...], ...]
    seed: int
    gamma: float

    def __post_init__(self):
        n = len(self.feature_names)
        frequency = np.asarray(self.frequencies)
        if (not n or len(set(self.feature_names)) != n or len(self.center) != n or len(self.scale) != n
                or frequency.ndim != 2 or not len(frequency) or frequency.shape[1] != n
                or not np.isfinite([*self.center, *self.scale, self.gamma]).all()
                or not np.isfinite(frequency).all() or min(self.scale) <= 0 or self.gamma <= 0):
            raise ValueError('Invalid Fourier map')

    @property
    def output_names(self):
        return tuple(f'rff_{trig}_{index:04d}' for trig in ('cos', 'sin') for index in range(len(self.frequencies)))

    def transform(self, values):
        x = np.asarray(values, float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid Fourier input')
        z = (x-self.center)/self.scale
        frequencies = np.asarray(self.frequencies)
        if len(z) == 0:
            return np.empty((0, 2*len(frequencies)))
        angles = np.array([frequencies@row for row in z])
        return np.c_[np.cos(angles), np.sin(angles)]/np.sqrt(len(frequencies))


def fit_fourier_map(values, sample_weights, *, feature_names, frequency_count=256, seed=20261006, scale_floor=.001):
    x, sw = np.asarray(values, float), np.asarray(sample_weights, float)
    if (x.ndim != 2 or not len(x) or x.shape[1] != len(feature_names) or sw.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(sw).all() or min(sw) <= 0
            or not isinstance(frequency_count, int) or frequency_count < 1 or not np.isfinite(scale_floor) or scale_floor <= 0):
        raise ValueError('Invalid Fourier fitting data')
    sw = sw/sw.sum()
    center = np.sum(x*sw[:, None], axis=0)
    scale = np.maximum(np.sqrt(np.sum((x-center)**2*sw[:, None], axis=0)), scale_floor)
    gamma = 1/x.shape[1]
    frequencies = np.random.default_rng(seed).normal(scale=np.sqrt(2*gamma), size=(frequency_count, x.shape[1]))
    return FourierMap(tuple(feature_names), tuple(center), tuple(scale), tuple(tuple(row) for row in frequencies), seed, gamma)


@dataclass(frozen=True)
class KernelRule:
    mapper: FourierMap
    readout: StableRule
    concatenate_input: bool = False

    def __post_init__(self):
        expected = (self.mapper.feature_names if self.concatenate_input else ()) + self.mapper.output_names
        if self.readout.feature_names != expected:
            raise ValueError('Kernel readout schema differs')

    @property
    def threshold(self):
        return self.readout.threshold

    def score(self, values):
        x = np.asarray(values, float)
        transformed = self.mapper.transform(x)
        if self.concatenate_input:
            transformed = np.c_[x, transformed]
        return self.readout.score(transformed)

    def payload(self):
        return {'schema': 1, 'kind': 'fourier_kernel_readout', 'mapper': asdict(self.mapper),
                'readout': self.readout.payload(), 'concatenate_input': self.concatenate_input}

    @property
    def fingerprint(self):
        return sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.payload(), indent=2)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding='utf-8'))
        if (value['schema'] != 1 or value['kind'] != 'fourier_kernel_readout'
                or value['readout']['schema'] != 1 or value['readout']['kind'] != 'paired_stability'):
            raise ValueError('Kernel schema differs')
        params = value['mapper']
        for key in ('feature_names', 'center', 'scale'):
            params[key] = tuple(params[key])
        params['frequencies'] = tuple(tuple(row) for row in params['frequencies'])
        head = value['readout']['rule']
        for key in ('feature_names', 'center', 'scale', 'weights'):
            head[key] = tuple(head[key])
        return cls(FourierMap(**params), StableRule(**head), value['concatenate_input'])
