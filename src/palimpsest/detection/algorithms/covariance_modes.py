"""Fit-only quadratic variance modes from matched-source centroids.

Class covariance spectra and quadratic feature expansion are classical. This
map does not enforce invariance; later source-risk fitting tests whether its
second-order signal remains useful across the registered processing views.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CovarianceModes:
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    directions: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    ridge: float
    shrinkage: float
    fit_manifest_sha256: str = ''

    def __post_init__(self):
        n, k = len(self.feature_names), len(self.eigenvalues)
        if (not n or len(set(self.feature_names)) != n or len(self.center) != n
                or len(self.scale) != n or not k or k > n
                or np.asarray(self.directions).shape != (n, k)
                or not all(np.isfinite(v).all() for v in
                           (self.center, self.scale, self.directions, self.eigenvalues, [self.ridge, self.shrinkage]))
                or min(self.scale) <= 0 or self.ridge <= 0 or not 0 <= self.shrinkage <= 1):
            raise ValueError('Invalid covariance-mode map')

    @property
    def quadratic_names(self):
        return tuple(f'covariance-mode/squared/{i}' for i in range(len(self.eigenvalues)))

    @property
    def output_names(self):
        return self.feature_names + self.quadratic_names

    def transform(self, values, *, quadratic_only=False):
        x = np.asarray(values, float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid covariance-mode input')
        z = (x-self.center)/self.scale
        directions = np.asarray(self.directions)
        # Preserve the same reductions for single-query and cached calls.
        q = np.array([(directions.T @ row)**2 for row in z])
        if not np.isfinite(q).all():
            raise ValueError('Nonfinite quadratic variance features')
        return q if quadratic_only else np.c_[x, q]

    def save(self, path: Path):
        path.write_text(json.dumps({'schema': 1, 'kind': 'covariance_modes', 'map': asdict(self)}, indent=2)+'\n')

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        if data['schema'] != 1 or data['kind'] != 'covariance_modes':
            raise ValueError('Covariance-mode schema differs')
        values = data['map']
        for key in ('feature_names', 'center', 'scale', 'eigenvalues'):
            values[key] = tuple(values[key])
        values['directions'] = tuple(tuple(row) for row in values['directions'])
        return cls(**values)


def fit_covariance_modes(values, labels, weights, sources, *, feature_names, count=32,
                         ridge=.1, shrinkage=.5, scale_floor=.001, manifest_sha=''):
    """Whiten centroid covariance, then retain largest absolute class differences.

    Centroids average only each fit source's registered views. Per-source mass
    comes from supplied weights; no additional images enter the moments.
    """
    x, y, sw, groups = np.asarray(values, float), np.asarray(labels), np.asarray(weights, float), np.asarray(sources)
    if (x.ndim != 2 or not len(x) or x.shape[1] != len(feature_names) or y.shape != (len(x),)
            or sw.shape != y.shape or groups.shape != y.shape or groups.dtype.kind not in 'iu'
            or set(y.tolist()) != {0, 1} or set(groups.tolist()) != set(range(int(groups.max())+1))
            or not 1 <= count <= x.shape[1] or not all(np.isfinite(v).all() for v in (x, sw))
            or min(sw) <= 0 or not np.isfinite([ridge, shrinkage, scale_floor]).all()
            or ridge <= 0 or scale_floor <= 0 or not 0 <= shrinkage <= 1):
        raise ValueError('Invalid covariance-mode fitting input')
    sw = sw/sw.sum()
    center = np.sum(sw[:, None]*x, axis=0)
    scale = np.maximum(np.sqrt(np.sum(sw[:, None]*(x-center)**2, axis=0)), scale_floor)
    z = (x-center)/scale
    centroids, masses, classes = [], [], []
    for source in range(int(groups.max())+1):
        selected = groups == source
        if len(set(y[selected].tolist())) != 1:
            raise ValueError('Conflicting labels within covariance source')
        mass = sw[selected].sum()
        centroids.append(np.sum(z[selected]*sw[selected, None], axis=0)/mass)
        masses.append(mass)
        classes.append(y[selected][0])
    centroids, masses, classes = np.asarray(centroids), np.asarray(masses), np.asarray(classes)
    covariances = []
    for label in (0, 1):
        selected = classes == label
        w = masses[selected]/masses[selected].sum()
        mean = np.sum(w[:, None]*centroids[selected], axis=0)
        residual = centroids[selected]-mean
        covariance = residual.T @ (w[:, None]*residual)
        covariances.append((covariance+covariance.T)/2)
    pooled = (covariances[0]+covariances[1])/2
    adjusted = (1-shrinkage)*pooled + shrinkage*np.diag(np.diag(pooled)) + ridge*np.eye(x.shape[1])
    spectrum, vectors = np.linalg.eigh(adjusted)
    if min(spectrum) <= 0:
        raise ValueError('Nonpositive pooled covariance')
    whitener = (vectors / np.sqrt(spectrum)) @ vectors.T
    error = float(np.linalg.norm(whitener @ adjusted @ whitener - np.eye(x.shape[1]), ord=np.inf))
    if error > 1e-7:
        raise ValueError('Covariance whitener residual exceeds tolerance')
    contrast = whitener @ (covariances[1]-covariances[0]) @ whitener
    eigenvalues, modes = np.linalg.eigh((contrast+contrast.T)/2)
    selected = np.argsort(-np.abs(eigenvalues), kind='stable')[:count]
    directions = whitener @ modes[:, selected]
    # Eigenvector sign is arbitrary; fix it before serialization.
    for i in range(count):
        pivot = int(np.argmax(np.abs(directions[:, i])))
        if directions[pivot, i] < 0:
            directions[:, i] *= -1
    mapper = CovarianceModes(tuple(feature_names), tuple(center), tuple(scale),
        tuple(tuple(row) for row in directions), tuple(eigenvalues[selected]), ridge, shrinkage, manifest_sha)
    return mapper, {'fit_records': len(x), 'fit_sources': len(centroids), 'dimensions': x.shape[1],
        'modes': count, 'minimum_pooled_eigenvalue': float(min(spectrum)), 'whitener_residual_inf': error,
        'selected_class_contrast_eigenvalues': eigenvalues[selected].tolist(), 'source_centroids_fit_only': True}
