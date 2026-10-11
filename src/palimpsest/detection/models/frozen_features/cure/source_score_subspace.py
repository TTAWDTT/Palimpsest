"""Compact source-risk readout over three supervised discriminant directions.

All bases and the final head are fitted on the same training-role rows. Validate
the whole pipeline in source CV; this is not out-of-fold Super Learner training.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json

import numpy as np

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
from palimpsest.detection.algorithms.readouts.consistent_source_risk import (
    fit_consistent_source_risk,
)
from palimpsest.detection.algorithms.csp_readout import CSPMap, CSPRule, fit_csp_source

BANK_NAMES = (
    "discriminant/mean",
    "discriminant/source_consistency",
    "discriminant/csp_source",
)


def stable_from_payload(value):
    if value["schema"] != 1 or value["kind"] != "paired_stability":
        raise ValueError("Invalid bank base payload")
    params = dict(value["rule"])
    for key in ("feature_names", "center", "scale", "weights"):
        params[key] = tuple(params[key])
    return StableRule(**params)


@dataclass(frozen=True)
class DiscriminantBank:
    feature_names: tuple[str, ...]
    mean: StableRule
    consistent: StableRule
    csp: CSPRule

    def __post_init__(self):
        prefix = self.feature_names[: self.csp.mapper.prefix_size]
        if (
            self.csp.mapper.feature_names != self.feature_names
            or self.mean.feature_names != prefix
            or self.consistent.feature_names != prefix
        ):
            raise ValueError("Discriminant bank schema differs")

    def transform(self, values):
        x = np.asarray(values, float)
        if (
            x.ndim != 2
            or x.shape[1] != len(self.feature_names)
            or not np.isfinite(x).all()
        ):
            raise ValueError("Invalid discriminant bank input")
        prefix = x[:, : self.csp.mapper.prefix_size]
        return np.c_[
            self.mean.score(prefix), self.consistent.score(prefix), self.csp.score(x)
        ]


@dataclass(frozen=True)
class ScoreSubspaceRule:
    bank: DiscriminantBank
    readout: StableRule

    def __post_init__(self):
        if self.readout.feature_names != BANK_NAMES:
            raise ValueError("Score subspace head schema differs")

    @property
    def threshold(self):
        return self.readout.threshold

    def score(self, values):
        return self.readout.score(self.bank.transform(values))

    def save(self, path):
        value = {
            "schema": 1,
            "kind": "source_score_subspace",
            "feature_names": self.bank.feature_names,
            "mean": self.bank.mean.payload(),
            "consistent": self.bank.consistent.payload(),
            "csp_mapper": asdict(self.bank.csp.mapper),
            "csp_head": self.bank.csp.readout.payload(),
            "readout": self.readout.payload(),
        }
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(path.read_text())
        if value["schema"] != 1 or value["kind"] != "source_score_subspace":
            raise ValueError("Invalid score subspace payload")
        params = value["csp_mapper"]
        params["feature_names"] = tuple(params["feature_names"])
        params["filters"] = tuple(tuple(row) for row in params["filters"])
        csp = CSPRule(CSPMap(**params), stable_from_payload(value["csp_head"]))
        bank = DiscriminantBank(
            tuple(value["feature_names"]),
            stable_from_payload(value["mean"]),
            stable_from_payload(value["consistent"]),
            csp,
        )
        return cls(bank, stable_from_payload(value["readout"]))


class ScoreSubspaceFitter:
    """Cache exact training-array bases across the fixed final-strength grid."""

    def __init__(self, names, *, channels=32, filter_count=16, manifest_sha=""):
        self.names = tuple(names)
        self.channels = channels
        self.filter_count = filter_count
        self.manifest_sha = manifest_sha
        self.bases = {}
        self.builds = 0

    def key(self, x, y, w, s):
        digest = sha256()
        for name, value, dtype in (
            ("x", x, "<f8"),
            ("y", y, "<i8"),
            ("w", w, "<f8"),
            ("s", s, "<i8"),
        ):
            a = np.ascontiguousarray(value, dtype=dtype)
            digest.update(f"{name}:{a.shape}:{dtype}".encode())
            digest.update(a.tobytes())
        return digest.hexdigest()

    def fit(self, x, y, w, s, strength, wrong=None):
        x, y, w, s = (
            np.asarray(x, float),
            np.asarray(y),
            np.asarray(w, float),
            np.asarray(s),
        )
        key = self.key(x, y, w, s)
        if key not in self.bases:
            prefix = len(self.names) - self.channels * (self.channels + 1) // 2
            names = self.names[:prefix]
            common = dict(
                feature_names=names,
                ridge=0.01,
                scale_floor=0.001,
                maximum_iterations=2000,
                gradient_tolerance=1e-5,
                manifest_sha=self.manifest_sha,
            )
            mean, d0 = fit_source_risk(x[:, :prefix], y, w, s, temperature=0, **common)
            consistent, d1 = fit_consistent_source_risk(
                x[:, :prefix], y, w, s, strength=0.1, temperature=0.1, **common
            )
            csp, d2 = fit_csp_source(
                x,
                y,
                w,
                s,
                feature_names=self.names,
                strength=0,
                channels=self.channels,
                filter_count=self.filter_count,
                manifest_sha=self.manifest_sha,
            )
            bank = DiscriminantBank(self.names, mean, consistent, csp)
            self.bases[key] = bank, {"mean": d0, "consistent": d1, "csp": d2}
            self.builds += 1
        bank, basis_diagnostics = self.bases[key]
        head, diagnostic = fit_consistent_source_risk(
            bank.transform(x),
            y,
            w,
            s,
            strength=strength,
            consistency_groups=wrong,
            feature_names=BANK_NAMES,
            temperature=0.1,
            ridge=0.01,
            scale_floor=0.001,
            maximum_iterations=2000,
            gradient_tolerance=1e-5,
            manifest_sha=self.manifest_sha,
        )
        return ScoreSubspaceRule(bank, head), {
            **diagnostic,
            "training_arrays_sha256": key,
            "basis_diagnostics": basis_diagnostics,
            "basis_count": 3,
            "fit_scope": "Bases and score head use same supplied training rows;whole-pipeline held-source CV required",
        }


def calibrate_score_subspace(rule, views, calibrate):
    mapped = {
        key: (records, rule.bank.transform(values), labels)
        for key, (records, values, labels) in views.items()
    }
    head, diagnostic = calibrate(rule.readout, mapped)
    return replace(rule, readout=head), diagnostic
