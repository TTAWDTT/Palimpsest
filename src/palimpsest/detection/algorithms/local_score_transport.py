"""Finite matched-source score corrections; no physical inverse guarantee."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from palimpsest.io.hashing import file_sha256
from palimpsest.detection.algorithms.readouts.stable_rule import StableRule


@dataclass(frozen=True)
class LocalScoreTransport:
    base: StableRule
    supports: np.ndarray
    anchors: tuple[float, ...]
    metric_weights: tuple[float, ...]
    neighbors: int = 5
    threshold: float = 0.0

    def __post_init__(self):
        values = np.asarray(self.supports, dtype=float)
        if (values.ndim != 3 or not values.shape[1] or values.shape[2] != len(self.feature_names)
                or len(self.anchors) != len(values) or len(self.metric_weights) != len(self.feature_names)
                or not isinstance(self.neighbors, int) or not 1 <= self.neighbors <= len(values)):
            raise ValueError('Invalid local score transport dimensions/neighbors')
        if (not np.isfinite(values).all() or not np.isfinite(self.anchors).all()
                or not np.isfinite(self.metric_weights).all() or min(self.metric_weights) <= 0
                or not np.isfinite(self.threshold)):
            raise ValueError('Invalid local score transport values')
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, 'supports', values)
        scaled = (2 * values - 1) * np.sqrt(self.metric_weights)
        scaled.setflags(write=False)
        object.__setattr__(self, '_scaled', scaled)
        object.__setattr__(self, '_norms', np.sum(scaled ** 2, axis=2))
        scores = self.base.score(values.reshape(-1, len(self.feature_names))).reshape(values.shape[:2])
        object.__setattr__(self, '_support_scores', scores)

    @property
    def feature_names(self):
        return self.base.feature_names

    def score(self, values):
        x = np.asarray(values, dtype=float)
        original = self.base.score(x)
        output = []
        anchors = np.array(self.anchors)
        for query, initial in zip(x, original):
            query = (2 * query - 1) * np.sqrt(self.metric_weights)
            squared = self._norms + np.sum(query ** 2) - 2 * np.einsum(
                'svd,d->sv', self._scaled, query, optimize=False)
            squared = np.maximum(0, squared)
            nearest_view = np.argmin(squared, axis=1)
            distances = squared[np.arange(len(squared)), nearest_view]
            # Stable ties follow the frozen support-source/view order.
            indices = np.argsort(distances, kind='stable')[:self.neighbors]
            correction = np.mean(anchors[indices] - self._support_scores[indices, nearest_view[indices]])
            output.append(float(initial + correction))
        return np.asarray(output)

    def save(self, path: Path):
        blob = path.with_suffix('.npz')
        np.savez_compressed(blob, supports=self.supports)
        value = {'schema': 1, 'kind': 'local_score_transport', 'base': self.base.payload(),
                 'anchors': self.anchors, 'metric_weights': self.metric_weights,
                 'neighbors': self.neighbors, 'threshold': self.threshold,
                 'supports_file': blob.name, 'supports_sha256': file_sha256(blob)}
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.pop('schema') != 1 or value.pop('kind') != 'local_score_transport':
            raise ValueError('Local score transport schema differs')
        name = value.pop('supports_file')
        if Path(name).name != name:
            raise ValueError('Local score transport blob must be adjacent')
        blob = path.parent / name
        if file_sha256(blob) != value.pop('supports_sha256'):
            raise ValueError('Local score transport blob changed')
        base = value['base']
        if base['schema'] != 1 or base['kind'] != 'paired_stability':
            raise ValueError('Local score transport base schema differs')
        params = base['rule']
        for key in ('feature_names', 'center', 'scale', 'weights'):
            params[key] = tuple(params[key])
        value['base'] = StableRule(**params)
        with np.load(blob, allow_pickle=False) as data:
            value['supports'] = data['supports']
        for key in ('anchors', 'metric_weights'):
            value[key] = tuple(value[key])
        return cls(**value)


def matched_source_anchors(base, values, sources, original_mask):
    """Return equal six-view bags and exactly two original scores per source."""
    x, groups, mask = np.asarray(values, float), np.asarray(sources), np.asarray(original_mask, bool)
    if (x.ndim != 2 or x.shape[1] != len(base.feature_names) or groups.shape != (len(x),)
            or mask.shape != (len(x),) or groups.dtype.kind not in 'iu'):
        raise ValueError('Invalid matched source arrays')
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    if not members or any(len(m) != 6 or mask[m].sum() != 2 for m in members):
        raise ValueError('Matched anchors require six views and two originals per source')
    scores = base.score(x)
    bags = np.array([x[m] for m in members])
    anchors = tuple(float(scores[m[mask[m]]].mean()) for m in members)
    return bags, anchors
