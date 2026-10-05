"""Pixel-feature conditional readout with an explicitly learned nuisance gate.

The gate is a clipped ridge estimate of an extra JPEG operation, not a physical
channel identifier or a calibrated probability. The final detector never takes
condition names, file metadata, reference pixels or other images as input.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.contracts import Prediction
from .paired_stability import StableRule
from .residual_statistics.features import FEATURE_NAMES, extract_features


def conditional_basis(values, gate_values):
    """Return [x, g*x, g]; nonlinear only through the pixel-derived gate g."""
    x, g = np.asarray(values, float), np.asarray(gate_values, float)
    if (x.ndim != 2 or g.shape != (len(x),) or not np.isfinite(x).all()
            or not np.isfinite(g).all() or np.any((g < 0) | (g > 1))):
        raise ValueError('Invalid conditional basis')
    return np.column_stack((x, x*g[:, None], g))


def gate_values(gate, values):
    return np.clip((gate.score(values)+1)/2, 0, 1)


def basis_names(names):
    return (*names, *(f'gate_product/{n}' for n in names), 'nuisance_gate')


@dataclass(frozen=True)
class ConditionalRule:
    gate: StableRule
    readout: StableRule
    threshold: float = 0.0

    def __post_init__(self):
        if self.readout.feature_names != basis_names(self.gate.feature_names) or not np.isfinite(self.threshold):
            raise ValueError('Conditional rule schemas differ')

    @property
    def feature_names(self):
        return self.gate.feature_names

    @property
    def fit_manifest_sha256(self):
        return self.readout.fit_manifest_sha256

    def score(self, values):
        return self.readout.score(conditional_basis(values, gate_values(self.gate, values)))

    def payload(self):
        return {'schema': 1, 'kind': 'conditional_residual', 'threshold': self.threshold,
                'gate': self.gate.payload(), 'readout': self.readout.payload()}

    @property
    def fingerprint(self):
        return sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.payload(), indent=2)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        p = json.loads(path.read_text(encoding='utf-8'))
        if p.get('schema') != 1 or p.get('kind') != 'conditional_residual':
            raise ValueError('Conditional rule schema differs')
        rules = []
        for name in ('gate', 'readout'):
            nested = p[name]
            if nested.get('schema') != 1 or nested.get('kind') != 'paired_stability':
                raise ValueError('Conditional component schema differs')
            params = nested['rule']
            for key in ('feature_names', 'center', 'scale', 'weights'):
                params[key] = tuple(params[key])
            rules.append(StableRule(**params))
        return cls(*rules, threshold=p['threshold'])


class ConditionalDetector:
    name = 'conditional-residual-score'

    def __init__(self, rule):
        if rule.feature_names != FEATURE_NAMES:
            raise ValueError('Conditional residual feature schema differs')
        self.rule = rule
        self.rule_sha256 = rule.fingerprint

    def predict(self, image):
        start = perf_counter()
        f = extract_features(image)
        score = float(self.rule.score(f.values[None])[0])
        return Prediction(self.name, score, self.rule.threshold, 'statistical_score',
                          timing_ms={'preprocess_ms': f.preprocess_ms, 'statistics_ms': f.statistics_ms,
                                     'predict_ms': 1000*(perf_counter()-start)},
                          metadata={'rule_sha256': self.rule_sha256,
                                    'fit_manifest_sha256': self.rule.fit_manifest_sha256})
