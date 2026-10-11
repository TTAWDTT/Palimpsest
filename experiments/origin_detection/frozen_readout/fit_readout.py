"""Bounded conventional readout; no images, metadata or selection labels at fit."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class Readout:
    estimator: RandomForestClassifier
    feature_count: int
    threshold: float = 0.0

    def score(self, features):
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.feature_count or not np.isfinite(values).all():
            raise ValueError("Invalid readout feature matrix")
        return self.estimator.predict_proba(values)[:, 1] - 0.5


def fit_readout(view_data, *, seed=20261005, shuffled=False):
    """Equal total sample weight for each view; optional source-wise label null."""
    if len(view_data) != 8:
        raise ValueError("Expected eight registered fit views")
    arrays, labels, weights, source_labels, source_groups = [], [], [], {}, {}
    for key, (records, values, truth) in sorted(view_data.items()):
        values, truth = np.asarray(values, dtype=float), np.asarray(truth)
        if (values.ndim != 2 or len(values) != len(truth) or len(records) != len(truth)
                or not np.isfinite(values).all() or set(truth) != {0, 1}
                or sum(truth == 0) != sum(truth == 1)):
            raise ValueError("Fit view must be finite and class-balanced")
        group = key.rsplit("/", 1)[0]
        seen = set()
        for row, label in zip(records, truth):
            if row['role'] != 'fit' or row['src'] in seen:
                raise ValueError("Only unique fit-role sources may train readout")
            seen.add(row['src'])
            prior = source_labels.setdefault(row['src'], int(label))
            if prior != label or source_groups.setdefault(row['src'], group) != group:
                raise ValueError("Conflicting source fit identity")
        arrays.append(values)
        labels.append(truth)
        weights.append(np.full(len(truth), 1.0 / len(truth)))
    if len({a.shape[1] for a in arrays}) != 1:
        raise ValueError("Fit feature dimensions differ")
    if shuffled:
        rng = np.random.default_rng(seed)
        for group in sorted(set(source_groups.values())):
            sources = sorted(s for s, g in source_groups.items() if g == group)
            permuted = rng.permutation([source_labels[s] for s in sources])
            source_labels.update(zip(sources, permuted.tolist()))
        labels = [np.array([source_labels[r['src']] for r in records])
                  for _, (records, _, _) in sorted(view_data.items())]
    estimator = RandomForestClassifier(n_estimators=128, max_depth=6, min_samples_leaf=8,
                                       max_features="sqrt", bootstrap=False, n_jobs=1, random_state=seed)
    estimator.fit(np.concatenate(arrays), np.concatenate(labels), sample_weight=np.concatenate(weights))
    if list(estimator.classes_) != [0, 1]:
        raise ValueError("Readout class ordering changed")
    return Readout(estimator, arrays[0].shape[1])
