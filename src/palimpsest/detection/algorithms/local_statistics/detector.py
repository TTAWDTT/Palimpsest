"""Explicit median/MAD rule, with labeled calibration and no probability claim."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.contracts import Prediction
from .features import FAMILIES, FEATURE_NAMES, extract_features


@dataclass(frozen=True)
class StatisticsRule:
    family: str
    long_edge: int
    indices: tuple[int, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    directions: tuple[int, ...]
    threshold: float = 0.0
    clip: float = 5.0
    fit_manifest_sha256: str = ""

    def __post_init__(self):
        if self.family not in FAMILIES or self.long_edge not in (512, 1024):
            raise ValueError("Unknown family/preprocessing")
        if not all(len(v) == len(self.indices) for v in (self.center, self.scale, self.directions)):
            raise ValueError("Rule parameter dimensions differ")
        if len(set(self.indices)) != len(self.indices) or any(i not in FAMILIES[self.family] for i in self.indices):
            raise ValueError("Invalid rule feature indices")
        if not np.isfinite([*self.center, *self.scale, self.threshold, self.clip]).all():
            raise ValueError("Nonfinite rule parameters")
        if any(v <= 0 for v in self.scale) or self.clip <= 0 or any(v not in (-1, 1) for v in self.directions):
            raise ValueError("Invalid scale/directions/clip")

    def score(self, features):
        array = np.asarray(features, dtype=np.float64)
        if array.shape[-1] != len(FEATURE_NAMES) or not np.isfinite(array).all():
            raise ValueError("Invalid feature table")
        if not self.indices:
            return np.zeros(array.shape[:-1])
        normalized = (array[..., self.indices] - self.center) / self.scale
        return np.mean(np.clip(normalized, -self.clip, self.clip) * self.directions, axis=-1)

    def payload(self):
        return {"schema": 1, "feature_names": list(FEATURE_NAMES), "rule": asdict(self)}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.payload(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["schema"] != 1 or value["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Rule schema/features changed")
        parameters = value["rule"]
        for key in ("indices", "center", "scale", "directions"):
            parameters[key] = tuple(parameters[key])
        return cls(**parameters)


class LocalStatisticsDetector:
    def __init__(self, rule: StatisticsRule):
        self.rule = rule
        self.name = f"local-statistics-{rule.family}-{rule.long_edge}"

    def predict(self, image):
        start = perf_counter()
        features = extract_features(image, self.rule.long_edge, family=self.rule.family)
        score = float(self.rule.score(features.values))
        return Prediction(self.name, score, self.rule.threshold, "statistical_score",
                          timing_ms={"preprocess_ms": features.preprocess_ms,
                                     "statistics_ms": features.statistics_ms,
                                     "predict_ms": (perf_counter() - start) * 1000},
                          metadata={"rule_sha256": self.rule.fingerprint,
                                    "fit_manifest_sha256": self.rule.fit_manifest_sha256,
                                    "patches": str(features.patches)})
