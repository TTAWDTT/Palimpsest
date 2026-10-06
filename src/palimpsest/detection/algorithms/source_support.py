"""Fixed class distances to source bags in a frozen feature metric.

Nearest support distances are smooth in the metric, but physical processing
need not move features by a known radius or preserve sufficient class signal.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from palimpsest.io.hashing import file_sha256


@dataclass(frozen=True)
class SourceSupportRule:
    feature_names: tuple[str, ...]
    supports: np.ndarray
    labels: tuple[int, ...]
    metric_weights: tuple[float, ...]
    neighbors: int = 5
    threshold: float = 0.0
    fit_manifest_sha256: str = ''

    def __post_init__(self):
        vectors = np.asarray(self.supports, dtype=float)
        d = len(self.feature_names)
        if (not d or len(set(self.feature_names)) != d or vectors.ndim != 3
                or not vectors.shape[1] or vectors.shape[2] != d
                or len(self.labels) != len(vectors) or set(self.labels) != {0, 1}
                or len(self.metric_weights) != d or not isinstance(self.neighbors, int)
                or self.neighbors < 1 or min(self.labels.count(c) for c in (0, 1)) < self.neighbors):
            raise ValueError('Invalid source-support dimensions/classes/neighbors')
        if (not np.isfinite(vectors).all() or not np.isfinite(self.metric_weights).all()
                or min(self.metric_weights) <= 0 or not np.isfinite(self.threshold)):
            raise ValueError('Invalid source-support metric/values')
        vectors = vectors.copy()
        vectors.setflags(write=False)
        object.__setattr__(self, 'supports', vectors)
        scaled = vectors * np.sqrt(self.metric_weights)
        scaled.setflags(write=False)
        object.__setattr__(self, '_scaled', scaled)
        object.__setattr__(self, '_norms', np.sum(scaled ** 2, axis=2))

    def score(self, values):
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid source-support query')
        scores = []
        labels = np.asarray(self.labels)
        # Always use the same per-query contraction, independent of batch length.
        for query in x:
            query = query * np.sqrt(self.metric_weights)
            squared = self._norms + np.sum(query ** 2) - 2 * np.einsum(
                'svd,d->sv', self._scaled, query, optimize=False)
            distances = np.sqrt(np.maximum(0, squared.min(axis=1)))
            class_distance = [np.partition(distances[labels == c], self.neighbors - 1)
                              [:self.neighbors].mean() for c in (0, 1)]
            scores.append(float(class_distance[0] - class_distance[1]))
        return np.asarray(scores)

    def save(self, path: Path):
        blob = path.with_suffix('.npz')
        np.savez_compressed(blob, supports=self.supports)
        value = {'schema': 1, 'kind': 'source_support', 'feature_names': self.feature_names,
                 'labels': self.labels, 'metric_weights': self.metric_weights,
                 'neighbors': self.neighbors, 'threshold': self.threshold,
                 'fit_manifest_sha256': self.fit_manifest_sha256,
                 'supports_file': blob.name, 'supports_sha256': file_sha256(blob)}
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.pop('schema') != 1 or value.pop('kind') != 'source_support':
            raise ValueError('Source-support artifact schema differs')
        name = value.pop('supports_file')
        if Path(name).name != name:
            raise ValueError('Source-support artifact must be adjacent')
        blob = path.parent / name
        if file_sha256(blob) != value.pop('supports_sha256'):
            raise ValueError('Source-support artifact changed')
        with np.load(blob, allow_pickle=False) as data:
            value['supports'] = data['supports']
        for key in ('feature_names', 'labels', 'metric_weights'):
            value[key] = tuple(value[key])
        return cls(**value)


def fit_source_support(values, labels, sources, *, feature_names, metric_weights,
                       neighbors=5, centroid=False, manifest_sha=''):
    """Store equal-view source bags or their mean; no gradient or target fit."""
    x, y, groups = np.asarray(values, float), np.asarray(labels), np.asarray(sources)
    if (x.ndim != 2 or y.shape != (len(x),) or groups.shape != (len(x),)
            or set(y.tolist()) != {0, 1} or groups.dtype.kind not in 'iu'):
        raise ValueError('Invalid source-support fit arrays')
    members = [np.flatnonzero(groups == source) for source in np.unique(groups)]
    if not members or len({len(indices) for indices in members}) != 1:
        raise ValueError('Source-support views must be nonempty and equal')
    source_labels, bags = [], []
    for indices in members:
        if len(set(y[indices])) != 1:
            raise ValueError('Conflicting source-support labels')
        source_labels.append(int(y[indices][0]))
        bags.append(x[indices])
    supports = np.array(bags)
    if centroid:
        supports = supports.mean(axis=1, keepdims=True)
    return SourceSupportRule(tuple(feature_names), supports, tuple(source_labels), tuple(metric_weights),
                             neighbors, fit_manifest_sha256=manifest_sha)
