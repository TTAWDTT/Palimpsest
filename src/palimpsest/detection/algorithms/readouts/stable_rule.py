"""Weighted ridge scores penalizing paired processing changes in feature space.

This is an empirical consistency objective, not a physical invariance theorem.
Training accepts explicit paired deltas; inference accepts only pixel features.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StableRule:
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    strength: float
    ridge: float
    threshold: float = 0.0
    fit_manifest_sha256: str = ""

    def __post_init__(self):
        n = len(self.feature_names)
        if (
            not n
            or len(set(self.feature_names)) != n
            or any(len(v) != n for v in (self.center, self.scale, self.weights))
        ):
            raise ValueError("Invalid stable rule dimensions")
        if (
            not np.isfinite(
                [
                    *self.center,
                    *self.scale,
                    *self.weights,
                    self.bias,
                    self.strength,
                    self.ridge,
                    self.threshold,
                ]
            ).all()
            or min(self.scale) <= 0
            or self.strength < 0
            or self.ridge <= 0
        ):
            raise ValueError("Invalid stable rule parameters")

    def score(self, values):
        x = np.asarray(values, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != len(self.feature_names)
            or not np.isfinite(x).all()
        ):
            raise ValueError("Invalid stable score feature matrix")
        # Row-wise reduction has the same summation path in single and batch use.
        return (
            np.sum(((x - self.center) / self.scale) * self.weights, axis=1) + self.bias
        )

    def payload(self):
        return {"schema": 1, "kind": "paired_stability", "rule": asdict(self)}

    @property
    def fingerprint(self):
        return sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.payload(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["schema"] != 1 or value["kind"] != "paired_stability":
            raise ValueError("Stable rule schema differs")
        params = value["rule"]
        for name in ("feature_names", "center", "scale", "weights"):
            params[name] = tuple(params[name])
        return cls(**params)


def fit_stable_rule(
    values,
    labels,
    sample_weights,
    pair_deltas,
    pair_weights,
    *,
    feature_names,
    strength,
    ridge=0.1,
    scale_floor=0.001,
    manifest_sha="",
):
    """Solve E[(y-f)^2]+strength*E[(w·delta)^2]+ridge*||w||², y=±1.

    Means/scales are fit-only weighted moments. Deltas are in original feature
    units and normalized by the same scale, with no subtraction of their mean.
    """
    x, y = np.asarray(values, float), np.asarray(labels)
    sw, d, pw = (
        np.asarray(sample_weights, float),
        np.asarray(pair_deltas, float),
        np.asarray(pair_weights, float),
    )
    if (
        x.ndim != 2
        or not len(x)
        or x.shape[1] != len(feature_names)
        or y.shape != (len(x),)
        or set(y.tolist()) != {0, 1}
        or sw.shape != (len(x),)
        or d.ndim != 2
        or not len(d)
        or d.shape[1] != x.shape[1]
        or pw.shape != (len(d),)
    ):
        raise ValueError("Invalid paired stable training dimensions/labels")
    if (
        not all(np.isfinite(z).all() for z in (x, sw, d, pw))
        or min(sw) <= 0
        or min(pw) <= 0
        or not np.isfinite([strength, ridge, scale_floor]).all()
        or strength < 0
        or ridge <= 0
        or scale_floor <= 0
    ):
        raise ValueError("Invalid paired stable training values")
    sw = sw / sw.sum()
    pw = pw / pw.sum()
    center = np.sum(x * sw[:, None], axis=0)
    scale = np.maximum(
        np.sqrt(np.sum((x - center) ** 2 * sw[:, None], axis=0)), scale_floor
    )
    z = (x - center) / scale
    delta = d / scale
    target = 2 * y - 1
    bias = float(np.sum(target * sw))
    scatter = z.T @ (sw[:, None] * z)
    paired = delta.T @ (pw[:, None] * delta)
    normal = scatter + strength * paired + ridge * np.eye(x.shape[1])
    rhs = z.T @ (sw * (target - bias))
    weights = np.linalg.solve(normal, rhs)
    rule = StableRule(
        tuple(feature_names),
        tuple(center),
        tuple(scale),
        tuple(weights),
        bias,
        strength,
        ridge,
        fit_manifest_sha256=manifest_sha,
    )
    fit_loss = float(np.sum(sw * (target - rule.score(x)) ** 2))
    drift = float(np.sum(pw * (delta @ weights) ** 2))
    regularizer = float(ridge * np.sum(weights**2))
    objective = fit_loss + strength * drift + regularizer
    baseline = float(np.sum(sw * (target - bias) ** 2))
    residual = float(
        np.linalg.norm(normal @ weights - rhs) / max(np.linalg.norm(rhs), 1.0)
    )
    if residual > 1e-9 or objective > baseline + 1e-9:
        raise ValueError("Paired stability optimization consistency failed")
    return rule, {
        "fit_squared_loss": fit_loss,
        "paired_score_drift": drift,
        "ridge_penalty": regularizer,
        "objective": objective,
        "constant_baseline_loss": baseline,
        "normal_equation_residual": residual,
    }
