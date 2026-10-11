"""Finite-view maximum hinge over fit-local supervised score directions.

Uses the existing bank, LP solver and portable rule without changing their
interfaces. This is conventional robust classification, not physical invariance.
"""

from palimpsest.detection.algorithms.readouts.source_hinge import fit_source_hinge
from palimpsest.detection.models.frozen_features.cure.source_score_subspace import (
    BANK_NAMES,
    ScoreSubspaceFitter,
    ScoreSubspaceRule,
)


class ScoreMarginFitter:
    """Build each exact training-array bank once; refit only its margin head.

    Zero denotes the mean-view control with fixed L1 .001. Positive parameters
    are maximum-source hinge L1 penalties. Bases share the supplied fit rows;
    whole-pipeline source CV is required, rather than OOF stacking claims.
    """

    def __init__(self, names, *, channels=32, filter_count=16, manifest_sha=""):
        self.provider = ScoreSubspaceFitter(
            names,
            channels=channels,
            filter_count=filter_count,
            manifest_sha=manifest_sha,
        )
        self.manifest_sha = manifest_sha

    def fit(self, x, y, weights, sources, penalty, wrong=None):
        key = self.provider.key(x, y, weights, sources)
        if key not in self.provider.bases:
            # Existing provider constructs a bank through its public fitter.
            # Its auxiliary zero-logistic head is counted, but not deployed.
            self.provider.fit(x, y, weights, sources, 0)
        bank, base_diagnostics = self.provider.bases[key]
        head, diagnostic = fit_source_hinge(
            bank.transform(x),
            y,
            weights,
            sources if wrong is None else wrong,
            feature_names=BANK_NAMES,
            maximum_source=penalty != 0,
            penalty=penalty if penalty != 0 else 0.001,
            time_limit=120,
            scale_floor=0.001,
            manifest_sha=self.manifest_sha,
        )
        return ScoreSubspaceRule(bank, head), {
            **diagnostic,
            "training_arrays_sha256": key,
            "basis_diagnostics": base_diagnostics,
            "bank_initialization_auxiliary_logistic_head": True,
            "fit_scope": "Same supplied rows for bank and margin head;whole-pipeline source CV",
        }
