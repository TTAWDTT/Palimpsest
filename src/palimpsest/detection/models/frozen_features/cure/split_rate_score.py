"""Learn score directions and final rate readout on disjoint fit sources.

Both stages use fit-role data. Threshold and outer sources remain separate.
This is sample splitting, not the two-player constraint-generalization theorem.
"""

import hashlib
import json

import numpy as np

from palimpsest.detection.algorithms.directed_margin import processing_pairs
from palimpsest.detection.algorithms.source_partition import source_half_split
from palimpsest.detection.models.frozen_features.cure.quantile_score_consistency import (
    QuantileScoreFitter,
    QuantileScoreRule,
    TERM_NAMES,
)
from palimpsest.detection.algorithms.readouts.rate_penalty import (
    training_rate_panel,
    fit_rate_source_risk,
)


class SplitRateScoreFitter:
    def __init__(self, names, *, variants, manifest_sha="", complement=False, **kwargs):
        if not isinstance(complement, bool):
            raise ValueError("Split complement must be explicit boolean")
        self.provider = QuantileScoreFitter(names, manifest_sha=manifest_sha, **kwargs)
        self.variants = tuple(variants)
        self.manifest_sha = manifest_sha
        self.complement = complement
        self.templates = {}
        self.template_diagnostics = {}
        self.readout_calls = 0

    def fit(self, x, labels, weights, sources, strength, wrong=None, *, records=None):
        if records is None:
            raise ValueError("Split fitting requires explicit training records")
        y = np.asarray(labels)
        if y.ndim != 1 or not set(y.tolist()) <= {0, 1}:
            raise ValueError("Invalid split wrapper labels")
        y = y.astype(np.int64)
        x, w, s = np.asarray(x), np.asarray(weights), np.asarray(sources)
        processing_pairs(records, y, w, s, variants=self.variants)
        if wrong is not None and (
            np.asarray(wrong).shape != y.shape
            or np.asarray(wrong).dtype.kind not in "iu"
        ):
            raise ValueError("Invalid split wrong control")
        mask = source_half_split(records, y)
        if self.complement:
            mask = ~mask
        head_mask = ~mask
        _, basis_groups = np.unique(s[mask], return_inverse=True)
        _, head_groups = np.unique(s[head_mask], return_inverse=True)
        array_key = self.provider.provider.provider.key(x, y, w, s)
        partition = [
            (r["domain"], r["scene"], r["src"], r["condition"], r["variant"], bool(m))
            for r, m in zip(records, mask)
        ]
        key = hashlib.sha256(
            json.dumps(
                [array_key, partition], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if key not in self.templates:
            # Only basis sources determine directions and score standardization.
            template, diagnostic = self.provider.fit(
                x[mask], y[mask], w[mask], basis_groups, 0
            )
            self.templates[key] = template
            self.template_diagnostics[key] = diagnostic
        template = self.templates[key]
        head_records = tuple(
            row for row, selected in zip(records, head_mask) if selected
        )
        panel = training_rate_panel(
            head_records,
            y[head_mask],
            w[head_mask],
            head_groups,
            variants=self.variants,
        )
        head, diagnostic = fit_rate_source_risk(
            template.transform(x[head_mask]),
            y[head_mask],
            w[head_mask],
            head_groups,
            panel,
            strength=strength,
            feature_names=TERM_NAMES,
            temperature=0.1,
            ridge=0.01,
            scale_floor=0.001,
            maximum_iterations=2000,
            gradient_tolerance=1e-5,
            optimizer_ftol=0.0,
            manifest_sha=self.manifest_sha,
        )
        self.readout_calls += 1
        basis_keys = sorted(
            {(r["domain"], r["src"]) for r, selected in zip(records, mask) if selected}
        )
        head_keys = sorted({(r["domain"], r["src"]) for r in head_records})
        if set(basis_keys) & set(head_keys) or len(basis_keys) != len(head_keys):
            raise ValueError("Stage source separation failed")
        return QuantileScoreRule(
            template.bank, template.center, template.scale, head
        ), {
            **diagnostic,
            "training_arrays_sha256": array_key,
            "stage_template_sha256": key,
            "basis_count": 5,
            "mapped_dimensions": 20,
            "basis_diagnostics": self.template_diagnostics[key],
            "basis_source_keys": basis_keys,
            "readout_source_keys": head_keys,
            "basis_fit_records": int(mask.sum()),
            "basis_source_count": len(basis_keys),
            "input_fit_records": len(x),
            "input_source_count": len(basis_keys) + len(head_keys),
            "partition_seed": 20261008,
            "partition_complement": self.complement,
            "wrong_source_affects_objective": False,
            "fit_records_scope": "fit_records/sources are readout solver subset;input_* is whole supplied training context",
            "fit_scope": "Disjoint source halves;directions/map from basis half,final risk/rate penalty from other half",
        }

    def audit(self):
        provider = self.provider.audit()
        if len(self.templates) < provider["banks"]:
            raise ValueError("Split bank ledger differs")
        return {
            "passed": True,
            "banks": provider["banks"],
            "maps": provider["maps"],
            "stage_templates": len(self.templates),
            "provider": provider,
            "initialization_source_readout_calls": provider[
                "source_consistency_fit_calls"
            ],
            "split_readout_fit_calls": self.readout_calls,
            "scope": "Stage templates include member identity;numeric child banks may deduplicate identical basis arrays;nonzero heads also solve baseline then penalty",
        }
