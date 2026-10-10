"""Conventional degree-two score expansion with finite-view margin fitting.

Nine terms over three supervised directions. No new neural training or physical
invariance is implied; map moments and all bases require fit-local provenance.
"""

from dataclasses import asdict, dataclass, replace
import json

import numpy as np

from palimpsest.detection.algorithms.csp_readout import CSPMap, CSPRule
from palimpsest.detection.algorithms.readouts.source_hinge import fit_source_hinge
from palimpsest.detection.models.frozen_features.cure.source_score_subspace import (
    BANK_NAMES,
    DiscriminantBank,
    ScoreSubspaceFitter,
    stable_from_payload,
)
from palimpsest.detection.algorithms.readouts.stable_rule import StableRule

QUADRATIC_NAMES = BANK_NAMES + tuple(
    f"score_quadratic/{i}/{j}" for i in range(3) for j in range(i, 3)
)


def quadratic_terms(z):
    """Exact finite map with inner product z·u+(z·u)^2 before head scaling."""
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[1] != 3 or not np.isfinite(z).all():
        raise ValueError("Invalid quadratic score input")
    with np.errstate(over="ignore", invalid="ignore"):
        terms = np.c_[
            z,
            *(
                z[:, i] * z[:, j] * (1 if i == j else np.sqrt(2))
                for i in range(3)
                for j in range(i, 3)
            ),
        ]
    if not np.isfinite(terms).all():
        raise ValueError("Nonfinite quadratic terms")
    return terms


def _stable_fields(value):
    return stable_from_payload({"schema": 1, "kind": "paired_stability", "rule": value})


def _bank_fields(value):
    mapper = dict(value["csp"]["mapper"])
    mapper["feature_names"] = tuple(mapper["feature_names"])
    mapper["filters"] = tuple(tuple(row) for row in mapper["filters"])
    csp = CSPRule(CSPMap(**mapper), _stable_fields(value["csp"]["readout"]))
    return DiscriminantBank(
        tuple(value["feature_names"]),
        _stable_fields(value["mean"]),
        _stable_fields(value["consistent"]),
        csp,
    )


@dataclass(frozen=True)
class QuadraticScoreRule:
    bank: DiscriminantBank
    center: tuple[float, ...]
    scale: tuple[float, ...]
    readout: StableRule

    def __post_init__(self):
        if (
            len(self.center) != 3
            or len(self.scale) != 3
            or min(self.scale) <= 0
            or not np.isfinite([*self.center, *self.scale]).all()
            or self.readout.feature_names != QUADRATIC_NAMES
        ):
            raise ValueError("Invalid quadratic score rule")

    @property
    def threshold(self):
        return self.readout.threshold

    def transform(self, values):
        return quadratic_terms((self.bank.transform(values) - self.center) / self.scale)

    def score(self, values):
        return self.readout.score(self.transform(values))

    def save(self, path):
        path.write_text(
            json.dumps(
                {"schema": 1, "kind": "quadratic_score_margin", "rule": asdict(self)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        value = json.loads(path.read_text())
        if value["schema"] != 1 or value["kind"] != "quadratic_score_margin":
            raise ValueError("Invalid quadratic rule payload")
        fields = value["rule"]
        return cls(
            _bank_fields(fields["bank"]),
            tuple(fields["center"]),
            tuple(fields["scale"]),
            _stable_fields(fields["readout"]),
        )


class QuadraticMarginFitter:
    """Memoize identical fit-only banks/maps, then fit the existing hinge LP."""

    def __init__(self, names, *, channels=32, filter_count=16, manifest_sha=""):
        self.provider = ScoreSubspaceFitter(
            names,
            channels=channels,
            filter_count=filter_count,
            manifest_sha=manifest_sha,
        )
        self.maps = {}
        self.manifest_sha = manifest_sha

    def fit(self, x, y, weights, sources, penalty, wrong=None):
        key = self.provider.key(x, y, weights, sources)
        if key not in self.provider.bases:
            self.provider.fit(x, y, weights, sources, 0)
        bank, diagnostics = self.provider.bases[key]
        if key not in self.maps:
            values = bank.transform(x)
            w = np.asarray(weights, float)
            w = w / w.sum()
            center = np.sum(values * w[:, None], axis=0)
            scale = np.maximum(
                np.sqrt(np.sum((values - center) ** 2 * w[:, None], axis=0)), 0.001
            )
            self.maps[key] = tuple(center), tuple(scale)
        center, scale = self.maps[key]
        terms = quadratic_terms((bank.transform(x) - center) / scale)
        head, diagnostic = fit_source_hinge(
            terms,
            y,
            weights,
            sources if wrong is None else wrong,
            feature_names=QUADRATIC_NAMES,
            maximum_source=penalty != 0,
            penalty=penalty if penalty != 0 else 0.001,
            scale_floor=0.001,
            time_limit=120,
            manifest_sha=self.manifest_sha,
        )
        return QuadraticScoreRule(bank, center, scale, head), {
            **diagnostic,
            "training_arrays_sha256": key,
            "basis_diagnostics": diagnostics,
            "bank_initialization_auxiliary_logistic_head": True,
            "mapped_dimensions": 9,
            "fit_scope": "Bank/map moments/head supplied fit rows only;whole-pipeline source CV",
        }

    def audit(self):
        count = self.provider.builds
        if set(self.maps) != set(self.provider.bases):
            raise ValueError("Quadratic map cache differs")
        return {
            "passed": True,
            "banks": count,
            "base_fits": 3 * count,
            "auxiliary_logistic_head_fits": count,
            "maps": len(self.maps),
            "training_array_keys": sorted(self.maps),
            "scope": "Array-key execution accounting;not independence",
        }


def calibrate_quadratic_score(rule, views, calibrate):
    mapped = {
        key: (records, rule.transform(values), labels)
        for key, (records, values, labels) in views.items()
    }
    head, diagnostic = calibrate(rule.readout, mapped)
    return replace(rule, readout=head), diagnostic
