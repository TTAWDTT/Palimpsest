"""Explicit labeled calibration, signed frequency average and one global threshold."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.contracts import Prediction
from .features import FEATURE_NAMES, extract_features


@dataclass(frozen=True)
class OrdinalRule:
    mode: str
    directions: tuple[int, ...]
    threshold: float = 0.0
    fit_manifest_sha256: str = ""

    def __post_init__(self):
        if (self.mode not in ("original_agreement", "processed_agreement")
                or len(self.directions) != len(FEATURE_NAMES)
                or any(type(v) is not int or v not in (-1, 0, 1) for v in self.directions)
                or not np.isfinite(self.threshold)):
            raise ValueError("Invalid ordinal rule")

    def score(self, features):
        values = np.asarray(features, dtype=float)
        if values.ndim < 1 or values.shape[-1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
            raise ValueError("Invalid ordinal feature dimensions/values")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("Ordinal frequencies outside [0,1]")
        count = sum(v != 0 for v in self.directions)
        return values @ self.directions / count if count else np.zeros(values.shape[:-1])

    def payload(self):
        return {"schema": 1, "feature_names": list(FEATURE_NAMES), "rule": asdict(self)}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.payload(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["schema"] != 1 or value["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Ordinal rule schema/features changed")
        return cls(**{**value["rule"], "directions": tuple(value["rule"]["directions"])})


class OrdinalDetector:
    def __init__(self, rule):
        self.rule = rule
        self.name = f"ordinal-statistics-{rule.mode}"

    def predict(self, image):
        start = perf_counter()
        features = extract_features(image)
        return Prediction(self.name, float(self.rule.score(features.values)), self.rule.threshold,
                          "statistical_score", timing_ms={"preprocess_ms": features.preprocess_ms,
                                                          "statistics_ms": features.statistics_ms,
                                                          "predict_ms": (perf_counter() - start) * 1000},
                          metadata={"rule_sha256": self.rule.fingerprint,
                                    "fit_manifest_sha256": self.rule.fit_manifest_sha256})
