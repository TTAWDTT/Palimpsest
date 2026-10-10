"""Fit a bounded linear rule orthogonal to paired processing differences.

This is a labeled calibration in clipped feature space, not an invariance
claim for raw images. The existing extractor and prediction contract are used.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.contracts import Prediction
from .features import FEATURE_NAMES, extract_features


@dataclass(frozen=True)
class ProjectionRule:
    center: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    basis: tuple[tuple[float, ...], ...]
    removed_rank: int
    threshold: float = 0.0
    clip: float = 5.0
    long_edge: int = 512
    fit_manifest_sha256: str = ""

    def __post_init__(self):
        dimension = len(FEATURE_NAMES)
        if any(len(v) != dimension for v in (self.center, self.scale, self.weights)):
            raise ValueError("Projection dimensions differ from feature schema")
        if self.removed_rank not in (0, 1, 3) or len(self.basis) > self.removed_rank:
            raise ValueError("Invalid removed rank")
        if any(len(row) != dimension for row in self.basis):
            raise ValueError("Invalid nuisance basis dimensions")
        if self.long_edge != 512 or self.clip <= 0 or any(v <= 0 for v in self.scale):
            raise ValueError("Invalid projection preprocessing/scale")
        if not np.isfinite([*self.center, *self.scale, *self.weights, self.threshold, self.clip]).all():
            raise ValueError("Nonfinite projection parameters")
        basis = np.asarray(self.basis, dtype=float).reshape((-1, dimension))
        if not np.isfinite(basis).all():
            raise ValueError("Nonfinite nuisance basis")
        if not np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-8, rtol=0):
            raise ValueError("Nuisance basis is not orthonormal")
        if not np.allclose(basis @ self.weights, 0, atol=1e-8, rtol=0):
            raise ValueError("Weights are not orthogonal to nuisance basis")
        norm = np.linalg.norm(self.weights)
        if norm != 0 and not np.isclose(norm, 1, atol=1e-8, rtol=0):
            raise ValueError("Weights must be unit length or a declared zero rule")

    @property
    def family(self):
        return f"paired-projection-rank{self.removed_rank}"

    def score(self, features):
        array = np.asarray(features, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != len(FEATURE_NAMES) or not np.isfinite(array).all():
            raise ValueError("Invalid projection feature table")
        z = np.clip((array - self.center) / self.scale, -self.clip, self.clip)
        return z @ np.asarray(self.weights)

    def payload(self):
        return {"schema": 1, "kind": "paired_projection", "feature_names": list(FEATURE_NAMES),
                "rule": asdict(self)}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.payload(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value["schema"] != 1 or value["kind"] != "paired_projection"
                or value["feature_names"] != list(FEATURE_NAMES)):
            raise ValueError("Projection schema/features changed")
        parameters = value["rule"]
        for key in ("center", "scale", "weights"):
            parameters[key] = tuple(parameters[key])
        parameters["basis"] = tuple(tuple(v) for v in parameters["basis"])
        return cls(**parameters)


def fit_projection(paired_features, labels, removed_rank, *, clip=5.0, manifest_sha=""):
    """Input is (original, transfer, redigital) x aligned source x 17 features."""
    array, labels = np.asarray(paired_features, dtype=float), np.asarray(labels)
    if (array.ndim != 3 or array.shape[0] != 3 or array.shape[2] != len(FEATURE_NAMES)
            or array.shape[1] == 0 or labels.shape != (array.shape[1],)
            or set(labels.tolist()) != {0, 1} or not np.isfinite(array).all()):
        raise ValueError("Invalid paired features or source labels")
    if removed_rank not in (0, 1, 3) or not np.isfinite(clip) or clip <= 0:
        raise ValueError("Invalid rank/clip")
    pool = array.reshape((-1, array.shape[2]))
    center = np.median(pool, axis=0)
    scale = np.maximum(np.median(np.abs(pool - center), axis=0), 1e-8)
    z = np.clip((array - center) / scale, -clip, clip)
    differences = np.concatenate([z[1] - z[0], z[2] - z[0]])
    # No mean subtraction: a fixed process-induced shift is also nuisance.
    _, singular, right = np.linalg.svd(differences, full_matrices=False)
    active = int(np.sum(singular > max(float(singular[0]), 1.0) * 1e-10))
    basis = right[:min(removed_rank, active)]
    contrast = z[:, labels == 1].mean(axis=(0, 1)) - z[:, labels == 0].mean(axis=(0, 1))
    projected = contrast - basis.T @ (basis @ contrast)
    norm, original_norm = np.linalg.norm(projected), np.linalg.norm(contrast)
    weights = projected / norm if norm > max(original_norm, 1.0) * 1e-10 else np.zeros_like(projected)
    total_energy = float(np.sum(singular ** 2))
    rule = ProjectionRule(tuple(center), tuple(scale), tuple(weights),
                          tuple(tuple(row) for row in basis), removed_rank,
                          clip=clip, fit_manifest_sha256=manifest_sha)
    diagnostics = {"effective_removed_rank": len(basis), "difference_rank": active,
                   "singular_values": singular.tolist(),
                   "removed_difference_energy_fraction": float(np.sum(singular[:len(basis)] ** 2)) / total_energy
                   if total_energy else 0.0,
                   "retained_contrast_norm_fraction": float(norm / original_norm) if original_norm else 0.0,
                   "fit_contrast_norm": float(original_norm),
                   "zero_rule": not bool(np.any(weights)),
                   "scope": "geometry in clipped fit feature coordinates; not image invariance or causal identification"}
    return rule, diagnostics


class PairedProjectionDetector:
    def __init__(self, rule: ProjectionRule):
        self.rule = rule
        self.name = rule.family

    def predict(self, image):
        start = perf_counter()
        features = extract_features(image, self.rule.long_edge)
        score = float(self.rule.score(features.values))
        return Prediction(self.name, score, self.rule.threshold, "statistical_score",
                          timing_ms={"preprocess_ms": features.preprocess_ms,
                                     "statistics_ms": features.statistics_ms,
                                     "predict_ms": (perf_counter() - start) * 1000},
                          metadata={"rule_sha256": self.rule.fingerprint,
                                    "fit_manifest_sha256": self.rule.fit_manifest_sha256,
                                    "patches": str(features.patches)})
