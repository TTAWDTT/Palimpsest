"""Full covariance of matched view differences for conventional support distance."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from palimpsest.io.hashing import file_sha256
from .source_support import SourceSupportRule


def paired_scatter(values, weights, sources):
    x, w, groups = np.asarray(values, float), np.asarray(weights, float), np.asarray(sources)
    if (x.ndim != 2 or not len(x) or w.shape != (len(x),) or groups.shape != w.shape
            or groups.dtype.kind not in 'iu' or not np.isfinite(x).all()
            or not np.isfinite(w).all() or np.min(w) <= 0):
        raise ValueError('Invalid paired-scatter arrays')
    w = w / w.sum()
    covariance = np.zeros((x.shape[1], x.shape[1]))
    sizes = set()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        count = len(indices)
        sizes.add(count)
        if count < 2 or not np.allclose(w[indices], w[indices[0]], atol=1e-15, rtol=0):
            raise ValueError('Paired views need equal weights and at least two records')
        centered = x[indices] - x[indices].mean(axis=0)
        covariance += w[indices].sum() * 2 / (count - 1) * (centered.T @ centered)
    if len(sizes) != 1:
        raise ValueError('Paired view panel sizes differ')
    return (covariance + covariance.T) / 2


def fit_query_map(values, weights, sources, *, paired, ridge=.01, scale_floor=.001):
    x, w = np.asarray(values, float), np.asarray(weights, float)
    if (x.ndim != 2 or not len(x) or w.shape != (len(x),) or not np.isfinite(x).all()
            or not np.isfinite(w).all() or np.min(w) <= 0
            or not np.isfinite([ridge, scale_floor]).all() or min(ridge, scale_floor) <= 0):
        raise ValueError('Invalid paired-metric fit')
    w = w / w.sum()
    center = np.sum(w[:, None] * x, axis=0)
    scale = np.maximum(np.sqrt(np.sum(w[:, None] * (x - center) ** 2, axis=0)), scale_floor)
    z = (x - center) / scale
    covariance = paired_scatter(z, w, sources)  # Also validate complete panels for Euclidean control.
    regularizer = ridge * np.trace(covariance) / x.shape[1]
    if paired and (not np.isfinite(regularizer) or regularizer <= 0):
        raise ValueError('Zero or nonfinite paired covariance trace')
    if paired:
        eigen, vectors = np.linalg.eigh(covariance + regularizer * np.eye(x.shape[1]))
        if not np.isfinite(eigen).all() or eigen[0] <= 0:
            raise ValueError('Nonpositive paired covariance eigenvalue')
        transform = (vectors / np.sqrt(eigen)) @ vectors.T
        error = float(np.max(np.abs(transform @ (covariance + regularizer * np.eye(x.shape[1]))
                                    @ transform - np.eye(x.shape[1]))))
        if error > 1e-8:
            raise ValueError('Paired inverse-root numerical identity failed')
    else:
        transform, error = np.eye(x.shape[1]), 0.
    return center, scale, transform, {'paired': bool(paired), 'paired_scatter_trace': float(np.trace(covariance)),
        'regularizer': float(regularizer), 'inverse_root_max_error': error, 'dimensions': x.shape[1]}


def transform_rows(values, center, scale, transform):
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != len(center) or not np.isfinite(x).all():
        raise ValueError('Invalid paired-metric query')
    return np.array([((row - center) / scale) @ transform for row in x]).reshape((len(x), len(center)))


@dataclass(frozen=True)
class PairedMetricSupportRule:
    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    transform: np.ndarray
    support: SourceSupportRule
    threshold: float = 0.

    def __post_init__(self):
        d = len(self.feature_names)
        if not d or len(set(self.feature_names)) != d or len(self.support.feature_names) != d:
            raise ValueError('Invalid paired-metric schema')
        for name, shape in (('center', (d,)), ('scale', (d,)), ('transform', (d, d))):
            array = np.asarray(getattr(self, name), float)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError('Invalid paired-metric map')
            array = array.copy()
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if np.min(self.scale) <= 0 or not np.isfinite(self.threshold):
            raise ValueError('Invalid paired-metric scale/threshold')

    def score(self, values):
        return self.support.score(transform_rows(values, self.center, self.scale, self.transform))

    def save(self, path: Path):
        map_path = path.with_name(path.stem + '_map.npz')
        support_path = path.with_name(path.stem + '_supports.json')
        np.savez_compressed(map_path, center=self.center, scale=self.scale, transform=self.transform)
        self.support.save(support_path)
        value = {'schema': 1, 'kind': 'paired_metric_support', 'feature_names': self.feature_names,
            'threshold': self.threshold, 'map_file': map_path.name, 'map_sha256': file_sha256(map_path),
            'support_file': support_path.name, 'support_sha256': file_sha256(support_path)}
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.pop('schema') != 1 or value.pop('kind') != 'paired_metric_support':
            raise ValueError('Paired-metric artifact schema differs')
        files = {}
        for name in ('map', 'support'):
            adjacent = value.pop(name + '_file')
            if Path(adjacent).name != adjacent:
                raise ValueError('Paired-metric artifact must be adjacent')
            files[name] = path.parent / adjacent
            if file_sha256(files[name]) != value.pop(name + '_sha256'):
                raise ValueError('Paired-metric artifact changed')
        with np.load(files['map'], allow_pickle=False) as arrays:
            for name in ('center', 'scale', 'transform'):
                value[name] = arrays[name]
        value['support'] = SourceSupportRule.load(files['support'])
        value['feature_names'] = tuple(value['feature_names'])
        return cls(**value)
