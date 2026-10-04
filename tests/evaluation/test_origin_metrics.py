import pytest

from palimpsest.evaluation.classification import auc, evaluate
from palimpsest.evaluation.pairing import paired_change
from palimpsest.evaluation.timing import summarize_timing


def test_auc_ties_count_as_half_and_require_two_classes():
    assert auc([(0.0, 0), (0.0, 1)]) == 0.5
    assert auc([(-2.0, 0), (-1.0, 0), (1.0, 1), (2.0, 1)]) == 1.0
    with pytest.raises(ValueError, match="both classes"):
        auc([(0.0, 1)])


def test_zero_is_real_and_source_macro_is_not_image_weighted():
    rows = [
        {"src": "r1", "label": "REAL", "score": "0"},
        {"src": "r2", "label": "REAL", "score": "1"},
        *[{"src": "f1", "label": "FAKE", "score": "1"} for _ in range(3)],
        {"src": "f2", "label": "FAKE", "score": "0"},
    ]
    metrics = evaluate(rows, "score")
    assert metrics["real_accuracy_at_zero"] == 0.5
    assert metrics["fake_accuracy_at_zero"] == 0.75
    assert metrics["balanced_accuracy_at_zero"] == 0.625
    assert metrics["source_macro_balanced_accuracy_at_zero"] == 0.5


def test_conflicting_source_labels_are_rejected():
    rows = [{"src": "same", "label": label, "score": "1"} for label in ("REAL", "FAKE")]
    with pytest.raises(ValueError, match="Conflicting"):
        evaluate(rows, "score")


def test_pairing_uses_source_intersection_and_reports_unpaired_sources():
    original = {
        "r": {"label": "REAL", "score": "-1"},
        "f": {"label": "FAKE", "score": "1"},
        "missing": {"label": "FAKE", "score": "1"},
    }
    processed = {
        "r": {"label": "REAL", "score": "1"},
        "f": {"label": "FAKE", "score": "1"},
    }
    result = paired_change(original, processed)
    assert result["balanced_accuracy_change"] == -0.5
    assert result["source_macro_decision_flip_rate"] == 0.5
    assert result["unpaired_original_sources"] == 1
    processed["r"]["label"] = "FAKE"
    with pytest.raises(ValueError, match="label mismatch"):
        paired_change(original, processed)


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_latency_rejects_nonfinite_and_negative_values(value):
    row = {
        key: value
        for key in ("decode_preprocess_ms", "gpu_transfer_forward_ms", "end_to_end_ms")
    }
    with pytest.raises(ValueError, match="Invalid timing"):
        summarize_timing([row])
