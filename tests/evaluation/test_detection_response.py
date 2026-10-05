import pytest
from palimpsest.contracts import Origin, Prediction
from palimpsest.evaluation.detection import DetectionObservation, evaluate_detection
from palimpsest.evaluation.channel_fidelity import compare_detector_response


def test_metrics_apply_explicit_threshold_and_preserve_source_pairs():
    observations = []
    for condition, scores in (("original", (0.2, 0.8)), ("transfer", (0.6, 0.9))):
        for source, label, score in zip(
            ("real0", "ai0"), (Origin.NATURAL, Origin.AI), scores
        ):
            observations.append(
                DetectionObservation(
                    source, condition, label, Prediction("fixed", score, threshold=0.5)
                )
            )
    report = evaluate_detection(observations)
    assert report["conditions"]["original"]["balanced_accuracy_at_zero"] == 1
    assert report["conditions"]["transfer"]["balanced_accuracy_at_zero"] == 0.5
    with pytest.raises(ValueError, match="one observation"):
        evaluate_detection(observations + [observations[0]])


def test_channel_response_requires_exact_alignment_and_detects_failure_match():
    original = {"ai": Prediction("fixed", 1), "real": Prediction("fixed", -1)}
    real = {"ai": Prediction("fixed", -1), "real": Prediction("fixed", -2)}
    labels = {"ai": Origin.AI, "real": Origin.NATURAL}
    result = compare_detector_response(original, real, real, labels)
    assert result["score_change_mae"] == 0 and result["real_failure_recall"] == 1
    with pytest.raises(ValueError, match="identical source"):
        compare_detector_response(original, real, {"ai": real["ai"]}, labels)
