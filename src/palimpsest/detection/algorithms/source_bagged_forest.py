"""Randomized trees with source-level sampling and portable, finite inference.

Grouping controls bootstrap dependence, not unknown propagation invariance.
The exported rule needs NumPy only; sklearn is imported only for fitting.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def bootstrap_weights(groups, labels, domains, *, seed, source_level=True):
    """Equal domain/class mass; optionally draw whole matched-source bags."""
    groups, labels = np.asarray(groups), np.asarray(labels)
    domains = np.asarray(domains)
    if (groups.ndim != 1 or not len(groups) or labels.shape != groups.shape
            or domains.shape != groups.shape or set(labels.tolist()) != {0, 1}):
        raise ValueError('Invalid bootstrap inputs')
    bags = defaultdict(list)
    for i, group in enumerate(groups.tolist()):
        bags[group].append(i)
    for indices in bags.values():
        if len(set(zip(domains[indices].tolist(), labels[indices].tolist()))) != 1:
            raise ValueError('Conflicting source domain/class')
    strata = defaultdict(list)
    if source_level:
        for group, indices in sorted(bags.items()):
            strata[(domains[indices[0]], int(labels[indices[0]]))].append(group)
    else:
        for i, (domain, label) in enumerate(zip(domains, labels)):
            strata[(domain, int(label))].append(i)
    domain_names = sorted(set(domains.tolist()))
    if set(strata) != {(d, y) for d in domain_names for y in (0, 1)}:
        raise ValueError('Missing domain/class bootstrap stratum')
    output = np.zeros(len(groups), dtype=float)
    rng = np.random.default_rng(seed)
    for _, units in sorted(strata.items()):
        drawn = rng.choice(units, size=len(units), replace=True)
        mass = 1 / len(domain_names) / 2 / len(units)
        for unit in drawn:
            indices = bags[unit] if source_level else [unit]
            output[indices] += mass / len(indices)
    return output


@dataclass(frozen=True)
class TreeNodes:
    left: tuple[int, ...]
    right: tuple[int, ...]
    feature: tuple[int, ...]
    split: tuple[float, ...]
    fake_probability: tuple[float, ...]

    def validate(self, dimensions):
        count = len(self.left)
        if (not count or any(len(v) != count for v in
                            (self.right, self.feature, self.split, self.fake_probability))
                or not np.isfinite(self.split).all()
                or not np.isfinite(self.fake_probability).all()
                or min(self.fake_probability) < 0 or max(self.fake_probability) > 1):
            raise ValueError('Invalid tree node values')
        parents = np.zeros(count, dtype=int)
        for i, (left, right, feature) in enumerate(zip(self.left, self.right, self.feature)):
            if any(not isinstance(v, int) for v in (left, right, feature)):
                raise ValueError('Noninteger tree index')
            if left == right == -1:
                if feature != -2:
                    raise ValueError('Invalid leaf feature')
            elif (not 0 <= feature < dimensions or not i < left < count
                  or not i < right < count or left == right):
                raise ValueError('Invalid/cyclic tree edges')
            else:
                parents[left] += 1
                parents[right] += 1
        if parents[0] or np.any(parents[1:] != 1):
            raise ValueError('Disconnected/duplicate tree node')

    def score(self, x):
        nodes = np.zeros(len(x), dtype=int)
        left, right, feature = (np.asarray(v) for v in (self.left, self.right, self.feature))
        split = np.asarray(self.split)
        while True:
            active = np.flatnonzero(left[nodes] != -1)
            if not len(active):
                return np.asarray(self.fake_probability)[nodes]
            current = nodes[active]
            decisions = x[active, feature[current]] <= split[current]
            nodes[active] = np.where(decisions, left[current], right[current])

    def payload(self):
        return {k: getattr(self, k) for k in
                ('left', 'right', 'feature', 'split', 'fake_probability')}


@dataclass(frozen=True)
class SourceBaggedForest:
    feature_names: tuple[str, ...]
    trees: tuple[TreeNodes, ...]
    threshold: float = 0.0

    def __post_init__(self):
        if (not self.feature_names or len(set(self.feature_names)) != len(self.feature_names)
                or not self.trees or not np.isfinite(self.threshold)):
            raise ValueError('Invalid forest schema')
        for tree in self.trees:
            tree.validate(len(self.feature_names))

    def score(self, values):
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid forest query')
        # sklearn's tree predictor converts X to float32 before comparisons.
        if np.any(np.abs(x) > np.finfo(np.float32).max):
            raise ValueError('Forest query exceeds float32 range')
        x = x.astype(np.float32)
        output = np.zeros(len(x))
        for tree in self.trees:
            output += tree.score(x)
        return output / len(self.trees) - .5

    def save(self, path: Path):
        path.write_text(json.dumps({'schema': 1, 'kind': 'source_bagged_forest',
            'feature_names': self.feature_names, 'threshold': self.threshold,
            'trees': [t.payload() for t in self.trees]}, separators=(',', ':')) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value['schema'] != 1 or value['kind'] != 'source_bagged_forest':
            raise ValueError('Unknown forest schema')
        trees = tuple(TreeNodes(**{k: tuple(v) for k, v in t.items()}) for t in value['trees'])
        return cls(tuple(value['feature_names']), trees, value['threshold'])


def fit_source_forest(x, labels, groups, domains, *, feature_names, seed,
                      source_level=True, trees=128, maximum_depth=12, minimum_leaf=12):
    from sklearn.tree import ExtraTreeClassifier
    import sklearn

    x, labels = np.asarray(x, dtype=float), np.asarray(labels)
    if (x.ndim != 2 or x.shape[1] != len(feature_names) or len(x) != len(labels)
            or not np.isfinite(x).all() or trees < 1):
        raise ValueError('Invalid forest fit matrix')
    native, exported, sample_audits = [], [], []
    for index in range(trees):
        weights = bootstrap_weights(groups, labels, domains, seed=seed + index, source_level=source_level)
        tree = ExtraTreeClassifier(criterion='gini', max_depth=maximum_depth,
            min_samples_leaf=minimum_leaf, max_features='sqrt', random_state=seed + index)
        tree.fit(x, labels, sample_weight=weights)
        if not np.array_equal(tree.classes_, [0, 1]):
            raise ValueError('Forest class order differs')
        t = tree.tree_
        exported.append(TreeNodes(tuple(t.children_left.tolist()), tuple(t.children_right.tolist()),
            tuple(t.feature.tolist()), tuple(t.threshold.tolist()), tuple(t.value[:, 0, 1].tolist())))
        native.append(tree)
        mass = {str(d): float(weights[np.asarray(domains) == d].sum()) for d in sorted(set(domains))}
        sample_audits.append({'positive_records': int(np.count_nonzero(weights)),
                             'nodes': t.node_count, 'depth': t.max_depth, 'domain_mass': mass})
    rule = SourceBaggedForest(tuple(feature_names), tuple(exported))
    expected = np.zeros(len(x))
    for tree in native:
        expected += tree.predict_proba(x)[:, 1]
    expected = expected / trees - .5
    error = float(np.max(np.abs(rule.score(x) - expected)))
    if error > 1e-15:
        raise ValueError('Exported forest differs from sklearn probabilities')
    return rule, {'sklearn_version': sklearn.__version__, 'trees': trees,
        'maximum_depth': maximum_depth, 'minimum_leaf_records': minimum_leaf,
        'source_level': source_level, 'native_export_max_error': error, 'sampling': sample_audits}
