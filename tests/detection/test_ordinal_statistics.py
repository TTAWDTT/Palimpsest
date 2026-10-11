"""Independent pixel oracle and full CSV/screen controls written before real run."""

import csv
import json

import numpy as np
from PIL import Image
import pytest

from palimpsest.data.source_groups import exact_source_components
from palimpsest.detection.algorithms.ordinal_statistics.features import FEATURE_NAMES, gray_features, extract_features
from palimpsest.detection.algorithms.ordinal_statistics.detector import OrdinalDetector, OrdinalRule
from palimpsest.detection.files import predict_file
from palimpsest.io.tables import read_rows
from experiments.origin_detection.ordinal_statistics.fit_rules import run_screen, validate_rows
from experiments.origin_detection.ordinal_statistics.run_iteration import write_json
from experiments.origin_detection.ordinal_statistics.audit_sources import chimera_inventory


def pixel_oracle(gray):
    # Different implementation: recursive reflect indexing and classify from bits,
    # without the production lookup array, OpenCV or vectorized comparisons.
    def reflect(index, size):
        if size == 1:
            return 0
        while index < 0 or index >= size:
            index = -index if index < 0 else 2 * size - 2 - index
        return index
    out = []
    for r in (1, 2, 4, 8):
        counts, ties = [0] * 6, 0
        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                neighbors = [gray[reflect(y + dy, gray.shape[0]), reflect(x + dx, gray.shape[1])]
                             for dy, dx in ((-r, 0), (0, r), (r, 0), (0, -r))]
                bits = [int(v >= gray[y, x]) for v in neighbors]
                n = sum(bits)
                orbit = (0 if n == 0 else 1 if n == 1 else 4 if n == 3 else 5 if n == 4
                         else 3 if bits[0] == bits[2] else 2)
                counts[orbit] += 1
                ties += sum(v == gray[y, x] for v in neighbors)
        out.extend(c / gray.size for c in counts)
        out.append(ties / (4 * gray.size))
    return np.array(out)


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (9, 7)])
def test_independent_oracle_and_constant_control(shape):
    gray = np.full(shape, 7.0)
    expected = np.tile([0, 0, 0, 0, 0, 1, 1], 4)
    np.testing.assert_array_equal(gray_features(gray), expected)
    pattern = np.random.default_rng(99).integers(0, 5, shape)
    np.testing.assert_array_equal(gray_features(pattern), pixel_oracle(pattern))
    if pattern.size > 1:
        assert not np.array_equal(gray_features(pattern), expected)


def test_scoped_invariance_and_deliberate_quantization_break():
    gray = np.random.default_rng(25).integers(0, 12, (19, 13)).astype(float)
    value = gray_features(gray)
    np.testing.assert_array_equal(value, gray_features(gray**3 + 5))
    for rotated in (np.rot90(gray), gray[:, ::-1], gray[::-1, :]):
        np.testing.assert_array_equal(value, gray_features(rotated))
    assert not np.array_equal(value, gray_features(gray // 4))
    corrupted = value.copy()
    corrupted[0] += 0.001
    assert not np.array_equal(corrupted, pixel_oracle(gray))


def plant():
    rows = []
    for role in ("fit", "selection", "threshold"):
        for domain, scenes, conditions in (("rr", ("all",), ("original", "transfer", "redigital")),
                                           ("chimera", ("cat", "church", "horse"), ("original", "mac_iphone", "lg_blackfly"))):
            for scene in scenes:
                for label in (0, 1):
                    for i in range(6):
                        src = f"{domain}/{scene}/{role}/{label}/{i}"
                        for condition in conditions:
                            # Independent known distribution: only first radius two
                            # complementary bins have class contrast; other bins constant.
                            vector = np.tile([0, 0, 0, 0, 0, 1, 0], 4).astype(float)
                            vector[0], vector[5] = label, 1 - label
                            rows.append({"filename": src + "/" + condition, "src": src, "role": role,
                                         "label": "FAKE" if label else "REAL", "domain": domain, "scene": scene,
                                         "condition": condition, "sha256": "fixture-digest",
                                         **dict(zip(FEATURE_NAMES, vector.tolist()))})
    return rows


def config():
    return {"candidate_modes": ["original_agreement", "processed_agreement"], "minimum_fit_auc": 0.55,
            "minimum_processed_class_accuracy": 0.55, "selection_auc_lower_gate": 0.5,
            "seed": 20261005, "bootstrap_repetitions": 20}


def csv_roundtrip(tmp_path, rows):
    path = tmp_path / "features.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return read_rows(path)


def test_csv_full_screen_known_signal_and_serialization(tmp_path):
    rows = csv_roundtrip(tmp_path, plant())
    validate_rows(rows, rows)
    result, rules = run_screen(rows, config(), "fixture")
    assert result["chosen"] == "original_agreement"
    for mode, rule in rules.items():
        assert [i for i, d in enumerate(rule.directions) if d] == [0, 5]
        assert rule.directions[0] == 1 and rule.directions[5] == -1
        assert result["candidates"][mode]["gate_passed"]
        path = tmp_path / f"{mode}.json"
        rule.save(path)
        assert OrdinalRule.load(path) == rule
    write_json(tmp_path / "receipt.json", result)
    assert json.loads((tmp_path / "receipt.json").read_text())["chosen"] == result["chosen"]


def test_processed_direction_flip_and_null_refused(tmp_path):
    rows = plant()
    for row in rows:
        if row["scene"] == "horse" and row["condition"] == "lg_blackfly":
            row[FEATURE_NAMES[0]], row[FEATURE_NAMES[5]] = row[FEATURE_NAMES[5]], row[FEATURE_NAMES[0]]
    result, rules = run_screen(csv_roundtrip(tmp_path, rows), config(), "fixture")
    assert result["chosen"] is None
    assert not any(rules["processed_agreement"].directions)
    assert result["candidates"]["original_agreement"]["selection_views"]["chimera/horse/lg_blackfly"]["auc"] == 0
    rows = plant()
    for r in rows:
        for key in FEATURE_NAMES:
            r[key] = 1.0 if key.endswith("_four") else 0.0
    result, rules = run_screen(csv_roundtrip(tmp_path, rows), config(), "fixture")
    assert result["chosen"] is None
    assert all(np.array_equal(rule.score(np.zeros((2, 28))), [0, 0]) for rule in rules.values())


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "label", "sha", "nan", "source"])
def test_cache_corruption_clean_twin(tmp_path, corruption):
    original = csv_roundtrip(tmp_path, plant())
    assert validate_rows(original, original)
    broken = [r.copy() for r in original]
    if corruption == "duplicate":
        broken.append(broken[0].copy())
    elif corruption == "missing":
        broken.pop()
    elif corruption == "nan":
        broken[0][FEATURE_NAMES[0]] = "nan"
    else:
        broken[0][{"label": "label", "sha": "sha256", "source": "src"}[corruption]] = "broken"
    with pytest.raises(ValueError):
        validate_rows(broken, original)


def test_live_file_roundtrip_and_rule_corruption(tmp_path):
    image = np.random.default_rng(33).integers(0, 256, (31, 27, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(image).save(path)
    rule = OrdinalRule("original_agreement", (1,) + (0,) * 27)
    result = predict_file(OrdinalDetector(rule), path)
    assert result.prediction.score == float(rule.score(extract_features(image).values))
    assert result.end_to_end_ms >= result.decode_ms >= 0
    with pytest.raises(ValueError):
        OrdinalRule("original_agreement", (1,) * 27)
    with pytest.raises(ValueError):
        OrdinalRule("original_agreement", (1,) * 28, threshold=float("nan"))
    with pytest.raises(ValueError):
        write_json(tmp_path / "broken.json", {"score": float("nan")})
    assert not (tmp_path / "broken.json").exists()


def test_exact_components_transitive_and_no_false_merge():
    actual = exact_source_components({"a": {"x"}, "b": {"x", "y"}, "c": {"y"}, "d": {"z"}})
    assert actual == {"a": "a", "b": "a", "c": "a", "d": "d"}
    assert exact_source_components({"a": {"x"}, "d": {"x"}})["d"] != actual["d"]


def test_actual_image_digest_refusal(tmp_path, monkeypatch):
    # Added after the feature run: engineering regression only, not an
    # independent pre-run control or retroactive scientific certification.
    from palimpsest.io.hashing import file_sha256
    from experiments.origin_detection.ordinal_statistics import run_iteration as runner
    path = tmp_path / "source.png"
    Image.fromarray(np.full((5, 5, 3), 100, np.uint8)).save(path)
    row = {"filename": "fixture/source.png", "sha256": file_sha256(path), "width": "5", "height": "5"}
    monkeypatch.setattr(runner, "image_path", lambda r: path)
    runner.extract_inventory([row], tmp_path / "clean.csv")
    assert (tmp_path / "clean.csv").exists()
    Image.fromarray(np.full((5, 5, 3), 101, np.uint8)).save(path)
    with pytest.raises(ValueError, match="digest changed"):
        runner.extract_inventory([row], tmp_path / "broken.csv")
    assert not (tmp_path / "broken.csv").exists()


def test_chimera_audit_clean_overlap_and_label_failure():
    manifest, audit = [], []
    for scene in ("cat", "church", "horse"):
        for label in ("FAKE", "REAL"):
            for i in range(5):
                src = f"{scene}/{label}/{i}"
                for condition in ("stylegan2_orig", "recap_mac", "recap_monitor"):
                    r = {"src": src, "label": label, "condition": condition, "filename": src + "/" + condition}
                    manifest.append(r)
                    audit.append({**r, "sha256": r["filename"], "width": "5", "height": "5"})
    cfg = {"seed": 1, "minimum_chimera_sources_per_cell": 2, "chimera_role_fractions": [0.6, 0.2, 0.2]}
    inventory, summary = chimera_inventory(manifest, audit, set(), cfg)
    assert len(inventory) == 90 and summary["used_sources"] == 30
    grouped = {}
    for r in inventory:
        grouped.setdefault(r["src"], set()).add(r["role"])
    assert all(len(v) == 1 for v in grouped.values())
    _, excluded = chimera_inventory(manifest, audit, {audit[0]["sha256"]}, cfg)
    assert excluded["used_sources"] == 29
    bad = [r.copy() for r in audit]
    bad[0]["label"] = "REAL"
    with pytest.raises(ValueError):
        chimera_inventory(manifest, bad, set(), cfg)
