"""Detector-response agreement on strictly matched original/real/simulated inputs.

This evaluates one downstream consequence, not physical correctness or utility
for training. Fitting the simulator to these scores invalidates held-out claims.
"""

import numpy as np
from palimpsest.contracts import Origin, Prediction


def compare_detector_response(
    original: dict[str, Prediction],
    real: dict[str, Prediction],
    simulated: dict[str, Prediction],
    labels: dict[str, Origin],
) -> dict:
    if not original or not (
        original.keys() == real.keys() == simulated.keys() == labels.keys()
    ):
        raise ValueError("Require nonempty identical source keys in all four mappings")
    signatures = {
        (p.method, p.score_kind, p.threshold)
        for group in (original, real, simulated)
        for p in group.values()
    }
    if len(signatures) != 1 or any(not isinstance(v, Origin) for v in labels.values()):
        raise ValueError("Require one fixed detector protocol and valid origin labels")
    keys = sorted(original)
    real_delta = np.array([real[k].score - original[k].score for k in keys])
    sim_delta = np.array([simulated[k].score - original[k].score for k in keys])
    correct_original = {k for k in keys if original[k].origin == labels[k]}
    real_fail = {k for k in correct_original if real[k].origin != labels[k]}
    sim_fail = {k for k in correct_original if simulated[k].origin != labels[k]}
    return {
        "sources": len(keys),
        "score_change_mae": float(np.mean(np.abs(real_delta - sim_delta))),
        "no_change_score_mae": float(np.mean(np.abs(real_delta))),
        "score_change_pearson": float(np.corrcoef(real_delta, sim_delta)[0, 1])
        if np.std(real_delta) > 0 and np.std(sim_delta) > 0
        else None,
        "real_correct_to_wrong": len(real_fail),
        "simulated_correct_to_wrong": len(sim_fail),
        "matched_failures": len(real_fail & sim_fail),
        "real_failure_recall": len(real_fail & sim_fail) / len(real_fail)
        if real_fail
        else None,
        "scope": "matched fixed-detector response only; not physical validation or training utility",
    }
