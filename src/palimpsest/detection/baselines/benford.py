"""Portable inference for the fitted Benford RF baseline; no implicit fitting.

A trusted fitted sklearn forest can be exported once. NPZ stores numeric trees
without pickle or a runtime sklearn dependency. The old RR run saved no forest;
its metrics alone cannot instantiate this detector.
"""

from pathlib import Path
from time import perf_counter
import numpy as np
from palimpsest.contracts import RGBImage, Prediction
from .benford_features import image_features


class BenfordRFDetector:
    name = "Benford-RF adaptation"

    def __init__(self, *, roots, left, right, feature, threshold, leaf_ai):
        self.roots = np.asarray(roots, dtype=np.int64)
        self.left = np.asarray(left, dtype=np.int64)
        self.right = np.asarray(right, dtype=np.int64)
        self.feature = np.asarray(feature, dtype=np.int64)
        self.threshold = np.asarray(threshold, dtype=np.float64)
        self.leaf_ai = np.asarray(leaf_ai, dtype=np.float64)
        size = len(self.left)
        if (
            size == 0
            or self.roots.ndim != 1
            or len(self.roots) == 0
            or self.roots[0] != 0
            or np.any(np.diff(self.roots) <= 0)
            or self.roots[-1] >= size
            or any(
                a.shape != (size,)
                for a in (
                    self.left,
                    self.right,
                    self.feature,
                    self.threshold,
                    self.leaf_ai,
                )
            )
            or not np.isfinite(self.threshold).all()
            or not np.isfinite(self.leaf_ai).all()
            or np.any((self.leaf_ai < 0) | (self.leaf_ai > 1))
        ):
            raise ValueError("Invalid numeric forest arrays")
        for begin, end in zip(self.roots, [*self.roots[1:], size]):
            for node in range(int(begin), int(end)):
                leaf = self.left[node] == self.right[node] == -1
                if not leaf and not (
                    node < self.left[node] < end
                    and node < self.right[node] < end
                    and 0 <= self.feature[node] < 18
                ):
                    raise ValueError("Invalid tree topology or feature index")

    @classmethod
    def from_estimator(cls, forest):
        """Export only a fitted binary sklearn RF using the 18-feature contract."""
        if forest.n_features_in_ != 18 or not np.array_equal(forest.classes_, [0, 1]):
            raise ValueError("Expected 18 features and classes [natural=0, AI=1]")
        arrays = {
            key: [] for key in ("left", "right", "feature", "threshold", "leaf_ai")
        }
        roots = []
        offset = 0
        for estimator in forest.estimators_:
            tree = estimator.tree_
            roots.append(offset)
            arrays["left"].extend(
                np.where(tree.children_left >= 0, tree.children_left + offset, -1)
            )
            arrays["right"].extend(
                np.where(tree.children_right >= 0, tree.children_right + offset, -1)
            )
            arrays["feature"].extend(tree.feature)
            arrays["threshold"].extend(tree.threshold)
            values = tree.value[:, 0, :]
            arrays["leaf_ai"].extend(values[:, 1] / values.sum(axis=1))
            offset += tree.node_count
        return cls(roots=roots, **arrays)

    def save(self, path: Path) -> None:
        """Refuse overwrites; provenance belongs alongside the explicit fit record."""
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                schema=np.array(1),
                roots=self.roots,
                left=self.left,
                right=self.right,
                feature=self.feature,
                threshold=self.threshold,
                leaf_ai=self.leaf_ai,
            )

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as archive:
            if archive["schema"].item() != 1:
                raise ValueError("Unsupported Benford forest schema")
            return cls(
                **{
                    key: archive[key]
                    for key in (
                        "roots",
                        "left",
                        "right",
                        "feature",
                        "threshold",
                        "leaf_ai",
                    )
                }
            )

    def score_features(self, features: np.ndarray) -> float:
        """Strict float32 traversal matches sklearn RF predict_proba input casting."""
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (18,) or not np.isfinite(features).all():
            raise ValueError("Expected 18 finite Benford features")
        probabilities = []
        for root in self.roots:
            node = int(root)
            while self.left[node] != -1:
                node = int(
                    self.left[node]
                    if features[self.feature[node]] <= self.threshold[node]
                    else self.right[node]
                )
            probabilities.append(self.leaf_ai[node])
        return float(np.mean(probabilities) - 0.5)

    def predict(self, image: RGBImage) -> Prediction:
        start = perf_counter()
        features = image_features(image)
        prepared = perf_counter()
        score = self.score_features(features)
        return Prediction(
            self.name,
            score,
            score_kind="uncalibrated_RF_probability_minus_0.5",
            timing_ms={
                "preprocess": (prepared - start) * 1000,
                "forward": (perf_counter() - prepared) * 1000,
            },
            metadata={
                "protocol": "center square 256 bicubic; DCT fixed-Q95; 18 features"
            },
        )
