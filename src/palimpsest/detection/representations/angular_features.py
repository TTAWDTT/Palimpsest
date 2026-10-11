"""Fixed direction/radius coordinates, with no physical invariance assumption."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


def nonzero_vectors(values, dimensions):
    x = np.asarray(values, float)
    if x.ndim != 2 or not len(x) or x.shape[1] != dimensions or not np.isfinite(x).all():
        raise ValueError('Invalid angular input dimensions/values')
    norms = np.array([np.linalg.norm(row) for row in x])
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Angular input needs finite nonzero norms')
    return x, norms


@dataclass(frozen=True)
class AngularFeatures:
    feature_names: tuple[str, ...]
    include_radius: bool = False

    def __post_init__(self):
        if (not self.feature_names or len(set(self.feature_names)) != len(self.feature_names)
                or type(self.include_radius) is not bool):
            raise ValueError('Invalid angular map schema')

    @property
    def output_names(self):
        direction = tuple('direction/'+n for n in self.feature_names)
        return direction + (('log_radius',) if self.include_radius else ())

    def transform(self, values):
        x, norms = nonzero_vectors(values, len(self.feature_names))
        direction = x / norms[:, None]
        return np.c_[direction, np.log(norms)] if self.include_radius else direction

    def save(self, path: Path):
        path.write_text(json.dumps({'schema': 1, 'kind': 'angular_features', 'map': asdict(self)}, indent=2)+'\n')

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        if data['schema'] != 1 or data['kind'] != 'angular_features':
            raise ValueError('Angular map schema differs')
        return cls(tuple(data['map']['feature_names']), data['map']['include_radius'])


def paired_drift(reference, processed):
    """Feature-space radial decomposition; not an image-channel diagnosis."""
    a = np.asarray(reference, float)
    if a.ndim != 2:
        raise ValueError('Invalid drift reference')
    a, na = nonzero_vectors(a, a.shape[1])
    b, nb = nonzero_vectors(processed, a.shape[1])
    if a.shape != b.shape:
        raise ValueError('Paired drift row count differs')
    dot = np.sum(a*b, axis=1)
    scale = dot/(na*na)
    return {'norm_ratio': nb/na, 'cosine': dot/(na*nb), 'best_scalar': scale,
            'nonradial_relative': np.array([np.linalg.norm(v) for v in b-scale[:, None]*a])/nb}
