"""Fixed mean of two opposite source-disjoint learners, one numeric input.

Deterministic score averaging is neither accuracy averaging nor independent
validation. Each component retains its own disjoint basis/readout sources.
"""

from dataclasses import asdict, dataclass, replace
import json

import numpy as np

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.models.frozen_features.cure.quantile_score_consistency import (
    QuantileScoreRule,
)
from palimpsest.detection.models.frozen_features.cure.split_rate_score import (
    SplitRateScoreFitter,
)
from palimpsest.detection.models.frozen_features.cure.source_score_subspace import (
    stable_from_payload,
)
from palimpsest.detection.algorithms.readouts.paired_threshold import paired_threshold

MEAN_NAMES = ("split_component/forward", "split_component/complement")
METHOD = "fixed equal mean of complementary split rate heads"


@dataclass(frozen=True)
class ComplementarySplitRule:
    components: tuple[QuantileScoreRule, QuantileScoreRule]
    readout: StableRule

    def __post_init__(self):
        if (
            len(self.components) != 2
            or self.components[0].bank.feature_names
            != self.components[1].bank.feature_names
            or any(c.threshold != 0 for c in self.components)
            or self.readout.feature_names != MEAN_NAMES
            or self.readout.center != (0.0, 0.0)
            or self.readout.scale != (1.0, 1.0)
            or self.readout.weights != (0.5, 0.5)
            or self.readout.bias != 0
        ):
            raise ValueError("Invalid fixed complementary mean rule")

    @property
    def threshold(self):
        return self.readout.threshold

    def transform(self, values):
        return np.column_stack([rule.score(values) for rule in self.components])

    def score(self, values):
        return self.readout.score(self.transform(values))

    def save(self, path):
        payload = {
            "schema": 1,
            "kind": "complementary_split",
            "components": [c.to_payload() for c in self.components],
            "readout": asdict(self.readout),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(path.read_text())
        if value["schema"] != 1 or value["kind"] != "complementary_split":
            raise ValueError("Unknown complementary mean schema")
        head = stable_from_payload(
            {"schema": 1, "kind": "paired_stability", "rule": value["readout"]}
        )
        return cls(
            tuple(QuantileScoreRule.from_payload(c) for c in value["components"]), head
        )


class ComplementarySplitFitter:
    def __init__(self, names, *, variants, manifest_sha="", **kwargs):
        self.providers = tuple(
            SplitRateScoreFitter(
                names,
                variants=variants,
                manifest_sha=manifest_sha,
                complement=c,
                **kwargs,
            )
            for c in (False, True)
        )
        self.manifest_sha = manifest_sha
        self.fit_calls = 0

    def fit(self, x, labels, weights, sources, strength, wrong=None, *, records=None):
        members, details = [], []
        for provider in self.providers:
            rule, diagnostic = provider.fit(
                x, labels, weights, sources, strength, wrong, records=records
            )
            members.append(rule)
            details.append(diagnostic)
        first, second = details
        if (
            first["basis_source_keys"] != second["readout_source_keys"]
            or first["readout_source_keys"] != second["basis_source_keys"]
        ):
            raise ValueError("Component source halves are not complementary")
        mean = StableRule(
            MEAN_NAMES,
            (0.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            0.0,
            strength,
            0.01,
            fit_manifest_sha256=self.manifest_sha,
        )
        self.fit_calls += 1
        return ComplementarySplitRule(tuple(members), mean), {
            "fit_records": sum(d["fit_records"] for d in details),
            "sources": sum(d["sources"] for d in details),
            "input_fit_records": len(x),
            "input_source_count": len(set(sources)),
            "maximum_absolute_gradient": max(
                d["maximum_absolute_gradient"] for d in details
            ),
            "consistency_strength": strength,
            "basis_count": 10,
            "mapped_dimensions": 2,
            "readout_method": METHOD,
            "components": details,
            "gradient_scope": "Maximum component gradient on two separate nonconvex objectives,not ensemble hard-rate certificate",
            "fit_records_scope": "Sum of disjoint readout halves;two basis/readout pipelines,not one full-source head",
        }

    def audit(self):
        components = [provider.audit() for provider in self.providers]
        return {
            "passed": all(d["passed"] for d in components),
            "banks": sum(d["banks"] for d in components),
            "maps": sum(d["maps"] for d in components),
            "components": components,
            "ensemble_fit_calls": self.fit_calls,
            "scope": "Numeric bank/component/ensemble entries are different units",
        }


def calibrate_complementary(rule, views, *, variants):
    mapped = {
        key: (records, rule.transform(x), labels)
        for key, (records, x, labels) in views.items()
    }
    head, diagnostic = paired_threshold(rule.readout, mapped, variants=variants)
    return replace(rule, readout=head), diagnostic
