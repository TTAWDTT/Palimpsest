"""Two explicit six-dimensional calibrated decisions, without runtime condition labels."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.contracts import Prediction
from .features import DECISION_INDICES, FEATURE_NAMES, extract_features


@dataclass(frozen=True)
class PhaseRule:
    mode: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    prototypes: tuple[tuple[tuple[float, ...], ...], ...]
    view_names: tuple[str, ...]
    threshold: float = 0.0
    clip: float = 5.0
    fit_manifest_sha256: str = ""

    def __post_init__(self):
        prototype = np.asarray(self.prototypes)
        if (self.mode not in ("linear", "nearest_prototype") or len(self.center) != 6 or len(self.scale) != 6
                or prototype.shape != (8, 2, 6) or len(self.view_names) != 8 or len(set(self.view_names)) != 8
                or not np.isfinite([*self.center, *self.scale, self.threshold, self.clip]).all()
                or not np.isfinite(prototype).all() or any(s <= 0 for s in self.scale) or self.clip <= 0):
            raise ValueError("Invalid phase rule dimensions/parameters")

    def normalized(self, features):
        values = np.asarray(features, dtype=float)
        if (values.ndim < 1 or values.shape[-1] != len(FEATURE_NAMES)
                or not np.isfinite(values).all() or np.any(np.abs(values) > 1)):
            raise ValueError("Invalid phase feature table")
        return np.clip((values[..., DECISION_INDICES] - self.center) / self.scale, -self.clip, self.clip)

    def score(self, features):
        values = self.normalized(features)
        prototypes = np.asarray(self.prototypes)
        if self.mode == "linear":
            direction = (prototypes[:, 1] - prototypes[:, 0]).mean(axis=0)
            norm = np.linalg.norm(direction)
            return values @ (direction / norm) if norm > 1e-12 else np.zeros(values.shape[:-1])
        distance = np.mean((values[..., None, None, :] - prototypes) ** 2, axis=-1).min(axis=-2)
        return distance[..., 0] - distance[..., 1]

    def payload(self):
        return {"schema": 1, "feature_names": list(FEATURE_NAMES), "decision_indices": list(DECISION_INDICES), "rule": asdict(self)}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.payload(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["schema"] != 1 or value["feature_names"] != list(FEATURE_NAMES) or value["decision_indices"] != list(DECISION_INDICES):
            raise ValueError("Phase rule schema/features changed")
        p = value["rule"]
        return cls(**{**p, "center": tuple(p["center"]), "scale": tuple(p["scale"]),
                      "view_names": tuple(p["view_names"]),
                      "prototypes": tuple(tuple(tuple(v) for v in pair) for pair in p["prototypes"])})


def fit_rules(view_data, *, clip=5.0, manifest_sha=""):
    if len(view_data) != 8:
        raise ValueError("Registered eight processed views required")
    arrays, medians, scales = [], [], []
    for _, values, labels in view_data.values():
        array = np.asarray(values, dtype=float)
        if (array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES) or not np.isfinite(array).all()
                or np.any(np.abs(array) > 1) or len(labels) != len(array) or set(labels) != {0, 1}):
            raise ValueError("Invalid labeled phase fit view")
        selected = array[:, DECISION_INDICES]
        center = np.median(selected, axis=0)
        medians.append(center)
        scales.append(np.median(np.abs(selected - center), axis=0))
        arrays.append((selected, np.asarray(labels)))
    center, scale = np.mean(medians, axis=0), np.maximum(np.mean(scales, axis=0), 1e-8)
    prototypes = [tuple(tuple(np.mean(np.clip((a[y == label] - center) / scale, -clip, clip), axis=0).tolist())
                        for label in (0, 1)) for a, y in arrays]
    return {mode: PhaseRule(mode, tuple(center.tolist()), tuple(scale.tolist()), tuple(prototypes),
                            tuple(view_data), clip=clip, fit_manifest_sha256=manifest_sha)
            for mode in ("linear", "nearest_prototype")}


class PhaseDetector:
    def __init__(self, rule):
        self.rule, self.name = rule, f"phase-statistics-{rule.mode}"

    def predict(self, image):
        start = perf_counter()
        features = extract_features(image)
        return Prediction(self.name, float(self.rule.score(features.values)), self.rule.threshold, "statistical_score",
                          timing_ms={"preprocess_ms": features.preprocess_ms, "statistics_ms": features.statistics_ms,
                                     "predict_ms": (perf_counter() - start) * 1000},
                          metadata={"rule_sha256": self.rule.fingerprint, "patches": str(features.patches),
                                    "fit_manifest_sha256": self.rule.fit_manifest_sha256})
