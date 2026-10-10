"""Pre-run harmonic and CSV controls authored independently of phase code."""

import csv
import json

import cv2
import numpy as np
from PIL import Image
import pytest

from palimpsest.detection.algorithms.phase_statistics.features import patch_features, extract_features, FEATURE_NAMES
from palimpsest.detection.algorithms.phase_statistics.detector import PhaseRule, PhaseDetector
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from experiments.origin_detection.phase_statistics.fit_rules import run_screen
from experiments.origin_detection.phase_statistics import run_iteration as runner


def harmonic_plant():
    y, x = np.mgrid[:128, :128]
    gray = np.full((128, 128), 100.0)
    wave_numbers = ((2, 1), (7, 2), (15, 5))
    phases = ((0.3, 0.2), (0.4, 1.3), (0.1, -0.7))
    for (ky, kx), (first, second) in zip(wave_numbers, phases):
        position = 2 * np.pi * (ky * y + kx * x) / 128
        gray += 5 * np.cos(position + first) + 3 * np.cos(2 * position + second)
    return gray, [2 * first - second for first, second in phases]


def test_independent_harmonic_answers_and_phase_break():
    gray, theta = harmonic_plant()
    result = patch_features(gray).reshape(3, 4)
    expected = [[np.cos(t), np.cos(2 * t), np.cos(4 * t)] for t in theta]
    np.testing.assert_allclose(result[:, :3], expected, atol=1e-12, rtol=0)
    # Independent enumeration rather than the production frequency-grid helper.
    counts = [sum(lo**2 <= y*y + x*x < hi**2 for y in range(-64, 64) for x in range(-64, 64))
              for lo, hi in ((2, 6), (6, 12), (12, 24))]
    np.testing.assert_array_equal(result[:, 3], [2 / n for n in counts])
    changed = gray + 2 * np.cos(2 * np.pi * (14 * np.mgrid[:128, :128][0] + 4 * np.mgrid[:128, :128][1]) / 128 + 0.1)
    assert abs(patch_features(changed)[5] - result.ravel()[5]) > 0.01


def test_circular_invariance_and_boundary_counterexample():
    gray, _ = harmonic_plant()
    reference = patch_features(gray).reshape(3, 4)
    for image in (np.roll(gray, (13, -7), (0, 1)), gray * 3 + 29):
        np.testing.assert_allclose(patch_features(image), reference.ravel(), atol=1e-12, rtol=0)
    # Nonnegative symmetric displaced PSF with signed cosine frequency response.
    blurred = (np.roll(gray, 3, 0) + np.roll(gray, -3, 0)) / 2
    actual = patch_features(blurred).reshape(3, 4)
    np.testing.assert_allclose(actual[:, 1:], reference[:, 1:], atol=1e-12, rtol=0)
    assert abs(actual[1, 0] - reference[1, 0]) > 0.1
    asymmetric = (gray * 0.25 + np.roll(gray, 3, 1) * 0.75)
    assert np.max(np.abs(patch_features(asymmetric).reshape(3, 4)[:, 1:3] - reference[:, 1:3])) > 0.01
    kernel = np.zeros((7, 1))
    kernel[0, 0] = kernel[6, 0] = 0.5
    finite_boundary = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    assert np.max(np.abs(patch_features(finite_boundary) - reference.ravel())) > 0.01
    # Destroyed support is explicitly outside the theorem, not repaired with eps phase.
    np.testing.assert_array_equal(patch_features(np.ones((128, 128))), np.zeros(12))


@pytest.mark.parametrize("shape", [(1, 1, 3), (17, 8, 3), (128, 128, 3), (256, 256, 3)])
def test_constant_null_small_inputs(shape):
    values = extract_features(np.full(shape, 128, np.uint8)).values
    np.testing.assert_array_equal(values, np.zeros(12))
    with pytest.raises(ValueError):
        extract_features(np.full(shape, 128.0))
    with pytest.raises(ValueError):
        patch_features(np.full((128, 128), np.nan))


def classification_plant():
    rows = []
    mapping = {("rr", "all", "transfer"): 0, ("rr", "all", "redigital"): 1,
               ("chimera", "cat", "mac_iphone"): 2, ("chimera", "cat", "lg_blackfly"): 3,
               ("chimera", "church", "mac_iphone"): 4, ("chimera", "church", "lg_blackfly"): 5,
               ("chimera", "horse", "mac_iphone"): 6, ("chimera", "horse", "lg_blackfly"): 7}
    for role in ("fit", "selection", "threshold"):
        for domain, scenes, conditions in (("rr", ("all",), ("original", "transfer", "redigital")),
                                           ("chimera", ("cat", "church", "horse"), ("original", "mac_iphone", "lg_blackfly"))):
            for scene in scenes:
                for label in (0, 1):
                    for i in range(6):
                        src = f"{domain}/{scene}/{role}/{label}/{i}"
                        for condition in conditions:
                            processed_condition = condition if condition != "original" else conditions[1]
                            index = mapping[domain, scene, processed_condition]
                            sign = 1 if index < 4 else -1
                            vector = np.zeros(12)
                            vector[1] = sign * (2 * label - 1) / 4
                            vector[2] = -sign / 4
                            for variant in runner.VARIANTS:
                                rows.append({"src": src, "filename": src + "/" + condition, "label": "FAKE" if label else "REAL",
                                             "role": role, "domain": domain, "scene": scene, "condition": condition,
                                             "variant": variant, "sha256": "independent-fixture", **dict(zip(FEATURE_NAMES, vector.tolist()))})
    return rows


def config():
    return {"candidate_modes": ["linear", "nearest_prototype"], "z_clip": 5.0, "bootstrap_repetitions": 20,
            "seed": 20261005, "selection_auc_lower_gate": 0.5, "minimum_processed_class_accuracy": 0.55,
            "minimum_original_class_accuracy": 0.55, "minimum_reencoded_ba": 0.55, "maximum_reencoded_ba_drop": 0.05}


def serialized_plant(tmp_path):
    path = tmp_path / "features.csv"
    rows = classification_plant()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rows = read_rows(path)
    inventory = [r for r in rows if r["variant"] == "raw"]
    validate_feature_cache(rows, inventory, FEATURE_NAMES, variants=runner.VARIANTS)
    return rows, inventory


def test_full_csv_fitting_condition_flip_and_roundtrip(tmp_path):
    rows, _ = serialized_plant(tmp_path)
    result, rules = run_screen(rows, config(), "plant")
    assert result["chosen"] == "nearest_prototype"
    assert result["candidates"]["nearest_prototype"]["gate_passed"]
    assert not result["candidates"]["linear"]["gate_passed"]
    assert result["candidates"]["nearest_prototype"]["worst_processed_auc"] == 1
    for mode, rule in rules.items():
        path = tmp_path / f"{mode}.json"
        rule.save(path)
        assert PhaseRule.load(path) == rule
    runner.write_json(tmp_path / "receipt.json", result)
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))["chosen"] == "nearest_prototype"
    # No runtime metadata is passed to score; two image features determine the answer.
    actual = np.zeros((2, 12))
    actual[:, 1], actual[:, 2] = [-0.25, 0.25], -0.25
    score = rules["nearest_prototype"].score(actual)
    assert score[0] <= rules["nearest_prototype"].threshold < score[1]
    assert not (score[0] > rules["nearest_prototype"].threshold)
    assert np.array_equal(rules["linear"].score(actual), [0, 0])


def test_null_and_class_destroyed_refusal(tmp_path):
    rows, inventory = serialized_plant(tmp_path)
    for r in rows:
        for n in FEATURE_NAMES:
            r[n] = "0"
    validate_feature_cache(rows, inventory, FEATURE_NAMES, variants=runner.VARIANTS)
    result, rules = run_screen(rows, config(), "null")
    assert result["chosen"] is None
    for rule in rules.values():
        np.testing.assert_array_equal(rule.score(np.zeros((2, 12))), [0, 0])


@pytest.mark.parametrize("kind", ["duplicate", "missing", "nan", "label", "role", "src"])
def test_cache_corruption_clean_twin(tmp_path, kind):
    rows, inventory = serialized_plant(tmp_path)
    broken = [r.copy() for r in rows]
    if kind == "duplicate":
        broken.append(broken[0])
    elif kind == "missing":
        broken.pop()
    else:
        broken[0][FEATURE_NAMES[0] if kind == "nan" else kind] = "nan" if kind == "nan" else "broken"
    with pytest.raises(ValueError):
        validate_feature_cache(broken, inventory, FEATURE_NAMES, variants=runner.VARIANTS)


def test_actual_extraction_digest_dimension_benchmark_and_score_refusal(tmp_path, monkeypatch):
    rows, _ = serialized_plant(tmp_path)
    _, rules = run_screen(rows, config(), "fixture")
    path = tmp_path / "image.png"
    rgb = np.random.default_rng(33).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(path)
    row = {"filename": "fixture.png", "sha256": file_sha256(path), "width": "128", "height": "128"}
    monkeypatch.setattr(runner, "image_path", lambda r: path)
    runner.extract_inventory([row], tmp_path / "clean.csv")
    expected = float(rules["nearest_prototype"].score(extract_features(rgb).values))
    item = {"filename": "fixture.png", "path": path, "sha256": row["sha256"], "expected_score": expected}
    detector = PhaseDetector(rules["nearest_prototype"])
    assert benchmark_files(detector, [item], repeats=1)["summaries"]["all"]["images"] == 1
    with pytest.raises(ValueError, match="mismatch"):
        benchmark_files(detector, [{**item, "expected_score": expected + 1}], repeats=1)
    with pytest.raises(ValueError):
        runner.extract_inventory([{**row, "width": "1"}], tmp_path / "badsize.csv")
    Image.fromarray(np.zeros_like(rgb)).save(path)
    with pytest.raises(ValueError, match="SHA"):
        runner.extract_inventory([row], tmp_path / "badbytes.csv")
    with pytest.raises(ValueError, match="digest"):
        benchmark_files(detector, [item], repeats=1)
    assert not (tmp_path / "badsize.csv").exists() and not (tmp_path / "badbytes.csv").exists()
