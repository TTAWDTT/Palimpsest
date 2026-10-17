import json

import numpy as np
import pytest

from palimpsest.contracts import Origin, Prediction
from palimpsest.detection.algorithms.marginal_intermittency import (
    AXES,
    LEGACY_THRESHOLD,
    AxisCalibration,
    MIntermittencyDetector,
    extract_features,
    extract_luminance,
)


def test_affine_invariance_on_float_luminance_without_clipping():
    rng = np.random.default_rng(281)
    yy, xx = np.mgrid[:256, :256]
    y = 0.3 + 0.12 * np.sin(xx / 7) + 0.08 * np.cos(yy / 11)
    y += 0.04 * rng.standard_normal(y.shape)
    baseline = extract_luminance(y)
    for gain in (2.7, -2.7):
        changed = extract_luminance(gain * y + 4.1)
        np.testing.assert_allclose(
            list(changed["features"].values()),
            list(baseline["features"].values()),
            rtol=0,
            atol=1e-10,
        )
        assert (
            changed["qc"]["ess_r_valid_fraction"]
            == baseline["qc"]["ess_r_valid_fraction"]
        )
    assert baseline["qc"]["logmoment_cauchy_violations_below_minus_1e_10"] == 0


def test_constant_image_is_finite_but_flagged_not_silently_rejected():
    image = np.full((120, 240, 3), 128, dtype=np.uint8)
    result = MIntermittencyDetector().analyze(image)
    assert result["qc"]["degenerate"]
    assert all(value == 0 for value in result["features"].values())
    assert np.isfinite(result["score"])
    assert result["prediction"] == int(result["score"] >= 0)


def test_packaged_calibration_and_legacy_zero_tie_are_preserved():
    detector = MIntermittencyDetector()
    centers = {axis: p.center for axis, p in zip(AXES, detector.parameters)}
    score, standardized = detector.score_features(centers)
    assert score == 0.0 and list(standardized.values()) == [0.0, 0.0, 0.0, 0.0]
    assert np.isclose(sum(p.weight for p in detector.parameters), 1.0, rtol=0, atol=1e-12)
    for value in (-1.0, LEGACY_THRESHOLD, -0.0, 0.0, 1.0):
        prediction = Prediction("M", value, threshold=LEGACY_THRESHOLD)
        assert (prediction.origin == Origin.AI) == (value >= 0)
    assert detector.parameters[0].center == 0.011286373441984687
    assert detector.parameters[1].scale == 0.007320421409122827
    assert detector.parameters[2].center == 1.9943937115323118
    assert detector.parameters[3].scale == 0.008271678466483186


@pytest.mark.parametrize(
    "scale,direction,weight",
    [
        (0, 1, 0.25),
        (-1, 1, 0.25),
        (np.nan, 1, 0.25),
        (1, 0, 0.25),
        (1, np.inf, 0.25),
        (1, 1, 0),
        (1, 1, -0.1),
    ],
)
def test_calibration_rejects_invalid_numeric_parameters(scale, direction, weight):
    with pytest.raises(ValueError, match="calibration"):
        AxisCalibration(0, scale, direction, weight)


def test_custom_calibration_requires_exact_schema(tmp_path):
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps({"schema": 2, "features": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        MIntermittencyDetector(path)


def test_rgb_contract_rejects_float_or_empty_input():
    with pytest.raises(ValueError, match="uint8"):
        extract_features(np.zeros((256, 256, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="uint8"):
        extract_features(np.zeros((0, 256, 3), dtype=np.uint8))
