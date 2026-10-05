"""Use typed predictions in the existing classification and paired protocols."""

from dataclasses import dataclass
from collections import defaultdict
from palimpsest.contracts import Origin, Prediction
from .classification import evaluate
from .pairing import paired_change


@dataclass(frozen=True)
class DetectionObservation:
    source_group: str
    condition: str
    label: Origin
    prediction: Prediction


def evaluate_detection(
    observations: list[DetectionObservation], *, reference="original"
) -> dict:
    if not observations:
        raise ValueError("No observations")
    methods = {
        (o.prediction.method, o.prediction.score_kind, o.prediction.threshold)
        for o in observations
    }
    if len(methods) != 1:
        raise ValueError(
            "Do not mix methods, score semantics or thresholds in one evaluation"
        )
    conditions = defaultdict(dict)
    labels = {}
    for obs in observations:
        if (
            not obs.source_group
            or not obs.condition
            or not isinstance(obs.label, Origin)
        ):
            raise ValueError(
                "Each observation needs source, condition and original-content label"
            )
        if obs.source_group in labels and labels[obs.source_group] != obs.label:
            raise ValueError("Conflicting original-content labels for a source group")
        labels[obs.source_group] = obs.label
        if obs.source_group in conditions[obs.condition]:
            raise ValueError("Expected one observation per source and condition")
        conditions[obs.condition][obs.source_group] = {
            "src": obs.source_group,
            "label": "FAKE" if obs.label == Origin.AI else "REAL",
            "score": obs.prediction.margin,
        }
    metrics = {
        condition: evaluate(list(rows.values()), "score")
        for condition, rows in conditions.items()
    }
    changes = {}
    if reference in conditions:
        for condition, rows in conditions.items():
            if condition == reference:
                continue
            changes[condition] = paired_change(conditions[reference], rows)
    return {
        "method": observations[0].prediction.method,
        "threshold": observations[0].prediction.threshold,
        "threshold_application": "strict score > threshold",
        "conditions": metrics,
        "paired_changes": changes,
    }
