"""Portable binary feature forests; NumPy inference without sklearn or pickle."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureTree:
    left: tuple[int, ...]
    right: tuple[int, ...]
    feature: tuple[int, ...]
    cut: tuple[float, ...]
    probability_ai: tuple[float, ...]


@dataclass(frozen=True)
class ForestRule:
    feature_names: tuple[str, ...]
    trees: tuple[FeatureTree, ...]
    threshold: float = 0.0
    fit_manifest_sha256: str = ""
    _arrays: tuple = field(init=False, repr=False, compare=False)
    _depth: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        dimensions = len(self.feature_names)
        if (not dimensions or len(set(self.feature_names)) != dimensions or not self.trees
                or len(self.trees) > 1024 or not np.isfinite(self.threshold)):
            raise ValueError("Invalid forest dimensions/threshold")
        depth = 0
        for tree in self.trees:
            size = len(tree.left)
            if (not size or any(len(a) != size for a in (tree.right, tree.feature, tree.cut, tree.probability_ai))
                    or any(type(v) is not int for a in (tree.left, tree.right, tree.feature) for v in a)
                    or not np.isfinite([*tree.cut, *tree.probability_ai]).all()
                    or any(not 0 <= p <= 1 for p in tree.probability_ai)):
                raise ValueError("Invalid tree arrays/probabilities")
            incoming, depths = np.zeros(size, int), np.zeros(size, int)
            for node, (left, right, feature) in enumerate(zip(tree.left, tree.right, tree.feature)):
                if left == right == -1:
                    continue
                if (not 0 <= feature < dimensions or not node < left < size or not node < right < size
                        or left == right):
                    raise ValueError("Invalid forward tree edges/features")
                for child in (left, right):
                    incoming[child] += 1
                    depths[child] = depths[node] + 1
            if incoming[0] != 0 or np.any(incoming[1:] != 1) or depths.max() > 64:
                raise ValueError("Disconnected/shared/deep tree")
            depth = max(depth, int(depths.max()))
        size = max(len(t.left) for t in self.trees)
        arrays = []
        for key, dtype, fill in [('left', int, -1), ('right', int, -1), ('feature', int, 0),
                                  ('cut', float, 0), ('probability_ai', float, 0)]:
            array = np.full((len(self.trees), size), fill, dtype=dtype)
            for index, tree in enumerate(self.trees):
                array[index, :len(tree.left)] = getattr(tree, key)
            array.setflags(write=False)
            arrays.append(array)
        object.__setattr__(self, '_arrays', tuple(arrays))
        object.__setattr__(self, '_depth', depth)

    def score(self, features):
        # sklearn's tree input convention is float32 even when training on float64.
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names) or not np.isfinite(values).all():
            raise ValueError("Invalid forest input matrix")
        left, right, feature, cut, probability = self._arrays
        tree_index = np.arange(len(self.trees))[:, None]
        item_index = np.arange(len(values))[None, :]
        nodes = np.zeros((len(self.trees), len(values)), dtype=int)
        for _ in range(self._depth):
            a, b = left[tree_index, nodes], right[tree_index, nodes]
            branch = a >= 0
            selected = values[item_index, np.maximum(feature[tree_index, nodes], 0)]
            next_nodes = np.where(selected <= cut[tree_index, nodes], a, b)
            nodes = np.where(branch, next_nodes, nodes)
        return probability[tree_index, nodes].mean(axis=0) - 0.5

    def payload(self):
        return {'schema': 1, 'input_precision': 'float32', 'feature_names': list(self.feature_names),
                'trees': [asdict(t) for t in self.trees], 'threshold': self.threshold,
                'fit_manifest_sha256': self.fit_manifest_sha256}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.payload(), allow_nan=False) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value['schema'] != 1 or value['input_precision'] != 'float32':
            raise ValueError("Unknown forest schema/input precision")
        return cls(tuple(value['feature_names']),
                   tuple(FeatureTree(**{key: tuple(v) for key, v in t.items()}) for t in value['trees']),
                   value['threshold'], value['fit_manifest_sha256'])

    @classmethod
    def from_estimator(cls, estimator, feature_names, *, manifest_sha=''):
        if list(estimator.classes_) != [0, 1] or estimator.n_features_in_ != len(feature_names):
            raise ValueError("Estimator class/feature identity differs")
        trees = []
        for estimator_tree in estimator.estimators_:
            t = estimator_tree.tree_
            counts = t.value[:, 0, :]
            trees.append(FeatureTree(tuple(t.children_left.tolist()), tuple(t.children_right.tolist()),
                                     tuple(t.feature.tolist()), tuple(t.threshold.tolist()),
                                     tuple((counts[:, 1] / counts.sum(axis=1)).tolist())))
        return cls(tuple(feature_names), tuple(trees), fit_manifest_sha256=manifest_sha)
