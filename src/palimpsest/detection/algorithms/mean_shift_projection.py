"""Finite class-conditional mean constraints, without a domain theorem."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .paired_stability import StableRule


@dataclass(frozen=True)
class MeanShiftProjection:
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    directions: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        n = len(self.feature_names)
        q = np.asarray(self.directions)
        if (not n or len(set(self.feature_names)) != n or len(self.center) != n or len(self.scale) != n
                or q.ndim != 2 or q.shape[0] != n or q.shape[1] >= n
                or not all(np.isfinite(v).all() for v in (self.center, self.scale, q)) or min(self.scale) <= 0):
            raise ValueError('Invalid mean-shift map')
        if q.shape[1] and np.max(np.abs(q.T@q-np.eye(q.shape[1]))) > 1e-8:
            raise ValueError('Mean-shift directions not orthonormal')

    @property
    def output_names(self):
        return tuple('mean_shift/'+n for n in self.feature_names)

    def transform(self, values):
        x = np.asarray(values, float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid mean-shift input')
        z = (x-self.center)/self.scale
        q = np.asarray(self.directions)
        return np.array([row-q@(q.T@row) for row in z]).reshape(x.shape)

    def collapse(self, head):
        if head.feature_names != self.output_names:
            raise ValueError('Mean-shift head schema differs')
        v = np.asarray(head.weights)/head.scale
        q = np.asarray(self.directions)
        native = (v-q@(q.T@v))/self.scale
        bias = head.bias-float(np.asarray(head.center)@v)
        return StableRule(self.feature_names, self.center, tuple(np.ones(len(native))), tuple(native), bias,
                          head.strength, head.ridge, head.threshold, head.fit_manifest_sha256)

    def save(self, path: Path):
        path.write_text(json.dumps({'schema': 1, 'kind': 'mean_shift_projection', 'map': asdict(self)}, indent=2)+'\n')

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        if data['schema'] != 1 or data['kind'] != 'mean_shift_projection':
            raise ValueError('Mean-shift schema differs')
        values = data['map']
        for key in ('feature_names', 'center', 'scale'):
            values[key] = tuple(values[key])
        values['directions'] = tuple(tuple(row) for row in values['directions'])
        return cls(**values)


def fit_mean_shift(values, labels, weights, sources, panels, views, *, feature_names,
                   reference_view, expected_views, scale_floor=.001, singular_floor=1e-8):
    x, y, w, s, panel, view = map(np.asarray, (values, labels, weights, sources, panels, views))
    if (x.ndim != 2 or not len(x) or x.shape[1] != len(feature_names)
            or any(v.shape != (len(x),) for v in (y, w, s, panel, view)) or set(y.tolist()) != {0, 1}
            or s.dtype.kind not in 'iu' or not np.isfinite(x).all() or not np.isfinite(w).all()
            or min(w) <= 0 or not np.isfinite([scale_floor, singular_floor]).all()
            or scale_floor <= 0 or not 0 < singular_floor < 1 or reference_view not in expected_views
            or len(set(expected_views)) != len(expected_views) or set(view.tolist()) != set(expected_views)):
        raise ValueError('Invalid mean-shift fitting input')
    for source in np.unique(s):
        mask = s == source
        if (len(set(y[mask].tolist())) != 1 or len(set(panel[mask].tolist())) != 1
                or sorted(view[mask].tolist()) != sorted(expected_views)
                or not np.allclose(w[mask], w[mask][0], rtol=0, atol=1e-15)):
            raise ValueError('Conflicting/incomplete source panel')
    w = w/w.sum()
    center = np.sum(x*w[:, None], axis=0)
    scale = np.maximum(np.sqrt(np.sum((x-center)**2*w[:, None], axis=0)), scale_floor)
    z = (x-center)/scale
    shifts = []
    for p in sorted(set(panel.tolist())):
        for c in (0, 1):
            means = {}
            for v in expected_views:
                mask = (panel == p) & (y == c) & (view == v)
                if not mask.any():
                    raise ValueError('Missing mean-shift class/view')
                means[v] = np.sum(z[mask]*w[mask, None], axis=0)/w[mask].sum()
            shifts.extend(means[v]-means[reference_view] for v in expected_views if v != reference_view)
    shifts = np.array(shifts)
    _, singular, right = np.linalg.svd(shifts, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0]*singular_floor)) if singular[0] > 0 else 0
    directions = right[:rank].T
    # SVD unit columns may be a few ulps short. Re-normalize explicitly so a
    # known coordinate-axis projector also has the exact expected null action.
    if rank:
        directions = directions/np.linalg.norm(directions, axis=0)
    if rank >= x.shape[1]:
        raise ValueError('Mean shifts consume entire feature space')
    residual = np.max(np.abs(shifts-(shifts@directions)@directions.T))
    relative = float(residual/max(1., np.max(np.abs(shifts))))
    if relative > 1e-8:
        raise ValueError('Mean-shift projection residual failed')
    mapper = MeanShiftProjection(tuple(feature_names), tuple(center), tuple(scale),
                                 tuple(tuple(row) for row in directions))
    return mapper, {'fit_records': len(x), 'fit_sources': len(np.unique(s)), 'shift_vectors': len(shifts),
                    'removed_rank': rank, 'singular_values': singular.tolist(), 'singular_floor': singular_floor,
                    'relative_projection_residual': relative,
                    'scope': 'Finite class-conditional means;not individual or unseen process invariance'}
