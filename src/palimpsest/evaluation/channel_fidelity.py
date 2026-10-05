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
        "simulated_failure_precision": len(real_fail & sim_fail) / len(sim_fail)
        if sim_fail
        else None,
        "processed_decision_agreement": sum(
            real[k].origin == simulated[k].origin for k in keys
        )
        / len(keys),
        "scope": "matched fixed-detector response only; not physical validation or training utility",
    }


def paired_score_error_gain(
    real: dict[str, Prediction],
    candidate: dict[str, Prediction],
    comparator: dict[str, Prediction],
    labels: dict[str, Origin],
    *,
    seed: int,
    replicates: int = 2000,
) -> dict:
    """Positive gain means candidate better matches real scores than comparator.

    Resample whole source groups independently within the two origin classes.
    A detector's score units are only comparable within its fixed protocol.
    """
    if not real or not (
        real.keys() == candidate.keys() == comparator.keys() == labels.keys()
    ):
        raise ValueError("Require identical nonempty source keys")
    if (
        len(
            {
                (p.method, p.threshold, p.score_kind)
                for group in (real, candidate, comparator)
                for p in group.values()
            }
        )
        != 1
    ):
        raise ValueError("Require one fixed detector protocol")
    if replicates < 2 or any(
        not isinstance(label, Origin) for label in labels.values()
    ):
        raise ValueError("Require valid labels and at least two bootstrap replicates")
    groups = []
    for label in (Origin.NATURAL, Origin.AI):
        values = np.array(
            [
                abs(real[k].score - comparator[k].score)
                - abs(real[k].score - candidate[k].score)
                for k in sorted(labels)
                if labels[k] == label
            ]
        )
        if not len(values):
            raise ValueError("Require both origin classes")
        groups.append(values)
    rng = np.random.default_rng(seed)
    estimates = np.zeros(replicates)
    for values in groups:
        for start in range(0, replicates, 100):
            count = min(100, replicates - start)
            indices = rng.integers(len(values), size=(count, len(values)))
            estimates[start : start + count] += values[indices].mean(axis=1) / 2
    return {
        "sources": len(labels),
        "class_balanced_mae_gain": float(np.mean([v.mean() for v in groups])),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "seed": seed,
        "replicates": replicates,
        "positive_means": "candidate has lower paired score error than comparator",
    }
