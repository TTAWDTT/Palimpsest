"""Gaussian source-anchor scores with finite matched-source regularization."""

from dataclasses import dataclass, asdict
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import pdist


def gaussian_rows(values, anchors, width):
    x, centers = np.asarray(values, np.float64), np.asarray(anchors, np.float64)
    if (x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]
            or not len(centers) or not np.isfinite(x).all() or not np.isfinite(centers).all()
            or not np.isfinite(width) or width <= 0):
        raise ValueError('Invalid Gaussian anchor schema/width')
    # Rowwise arithmetic is identical for cached panels and one query. Avoid
    # cancellation from norm+dot expansions and batch-dependent BLAS summation.
    result = np.empty((len(x), len(centers)))
    for i, row in enumerate(x):
        difference = centers-row
        result[i] = np.exp(-np.sum(difference*difference, axis=1)/(2*width))
    return result


def source_centers(values, sources):
    x, groups = np.asarray(values, np.float64), np.asarray(sources)
    if x.ndim != 2 or groups.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('Invalid source-center panel')
    centers = np.stack([x[groups == g].mean(axis=0) for g in np.unique(groups)])
    distances = pdist(centers, metric='sqeuclidean')
    if not len(distances) or not np.isfinite(distances).all():
        raise ValueError('Missing finite source distances')
    width = float(np.median(distances[distances > 0])) if (distances > 0).any() else 0.
    if width <= 0:
        raise ValueError('Source anchors have no positive scale')
    return centers, width


def solve_panel(kernel, labels, weights, sources, anchor_kernel, *, strength, ridge=.01, nugget=1e-6):
    phi, y, w, groups, kaa = (np.asarray(a) for a in (kernel, labels, weights, sources, anchor_kernel))
    if (phi.ndim != 2 or y.shape != (len(phi),) or w.shape != y.shape or groups.shape != y.shape
            or kaa.shape != (phi.shape[1], phi.shape[1]) or set(y.tolist()) != {0, 1}
            or not np.isfinite(phi).all() or not np.isfinite(kaa).all() or not np.isfinite(w).all()
            or (w <= 0).any() or abs(w.sum()-1) > 1e-10
            or not np.isfinite([strength, ridge, nugget]).all() or strength < 0 or ridge <= 0 or nugget <= 0
            or not np.allclose(kaa, kaa.T, atol=1e-12, rtol=0)):
        raise ValueError('Invalid paired kernel fit panel')
    phi, w = phi.astype(np.float64), w.astype(np.float64)
    target = 2*y.astype(np.float64)-1
    center, target_center = np.sum(w[:, None]*phi, axis=0), float(w@target)
    centered = phi-center
    within = np.empty_like(phi)
    for g in np.unique(groups):
        mask = groups == g
        if len(set(y[mask].tolist())) != 1 or mask.sum() < 2:
            raise ValueError('Conflicting/incomplete class-source panel')
        within[mask] = phi[mask]-np.average(phi[mask], axis=0, weights=w[mask])
    gram = within.T@(w[:, None]*within)
    matrix = centered.T@(w[:, None]*centered)+strength*gram+ridge*kaa+nugget*np.eye(phi.shape[1])
    matrix = (matrix+matrix.T)/2
    right = centered.T@(w*(target-target_center))
    alpha = cho_solve(cho_factor(matrix, lower=True, check_finite=True), right)
    residual = float(np.max(np.abs(matrix@alpha-right)))/max(1., float(np.max(np.abs(right))))
    if not np.isfinite(alpha).all() or residual > 1e-8:
        raise ValueError('Paired kernel normal equation failed')
    bias = target_center-float(center@alpha)
    prediction = phi@alpha+bias
    return alpha, bias, {'relative_normal_residual': residual,
        'weighted_fit_squared_error': float(w@((prediction-target)**2)),
        'weighted_source_variance': float(alpha@gram@alpha),
        'strength': strength, 'ridge': ridge, 'nugget': nugget}


@dataclass(frozen=True)
class PairedKernelRule:
    feature_names: tuple[str, ...]
    anchors: tuple[tuple[float, ...], ...]
    width: float
    alpha: tuple[float, ...]
    bias: float
    threshold: float
    manifest_sha: str

    def score(self, values):
        x = np.asarray(values, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError('Kernel query dimensions differ')
        k = gaussian_rows(x, self.anchors, self.width)
        result = np.sum(k*np.asarray(self.alpha)[None, :], axis=1)+self.bias
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite kernel score')
        return result

    def save(self, path):
        path.write_text(json.dumps(asdict(self), separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding='utf-8'))
        data['feature_names'] = tuple(data['feature_names'])
        data['anchors'] = tuple(tuple(a) for a in data['anchors'])
        data['alpha'] = tuple(data['alpha'])
        rule = cls(**data)
        if len(rule.alpha) != len(rule.anchors) or not np.isfinite([rule.bias, rule.threshold]).all():
            raise ValueError('Kernel rule schema differs')
        rule.score(np.zeros((1, len(rule.feature_names))))
        return rule
