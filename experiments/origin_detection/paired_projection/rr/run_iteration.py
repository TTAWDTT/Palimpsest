"""Bounded second iteration using frozen features; no image reads or refits on holdout."""

from collections import defaultdict
import argparse
from datetime import datetime, timezone
import json
from time import perf_counter
import tomllib

import numpy as np

from palimpsest.contracts import Origin
from palimpsest.detection.algorithms.local_statistics.detector import StatisticsRule
from palimpsest.detection.algorithms.local_statistics.projection import fit_projection
from palimpsest.evaluation.cached import CachedRun, ExpectedImage, evaluate_cached_method
from palimpsest.evaluation.classification import evaluate
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import DATA_ROOT, REPO_ROOT, WORK_DIR
from experiments.origin_detection.robust_statistics.rr.evaluate_features import (
    candidate_auc, classification_for_role, matrix, threshold_for_rule, validate_feature_rows,
)
from experiments.origin_detection.robust_statistics.rr.protocol import (
    RESULT as FIRST, settings, verified_inventory, write_json,
)
from experiments.origin_detection.robust_statistics.rr.run_features import code_fingerprints

CONFIG = REPO_ROOT / "configs/evaluation/rr_paired_projection.toml"
OUTPUT = WORK_DIR / "robust_statistics/rr_paired_projection"


def paired_matrices(table, role, config):
    arrays, names, labels = [], None, None
    for condition in config["conditions"]:
        rows, values, current_labels = matrix(table, config["long_edge"], "raw", role, condition)
        current_names = [row["src"] for row in rows]
        if names is not None and (names != current_names or not np.array_equal(labels, current_labels)):
            raise ValueError("Paired source/label alignment differs")
        names, labels = current_names, current_labels
        arrays.append(values)
    return np.stack(arrays), labels


def score_drift(table, rule, role, config):
    arrays, labels = paired_matrices(table, role, config)
    scores = rule.score(arrays)
    return {condition: {label_name: {
        "sources": int(np.sum(labels == label)),
        "mean_score_change": float(np.mean((scores[index] - scores[0])[labels == label])),
        "mean_absolute_score_change": float(np.mean(np.abs(scores[index] - scores[0])[labels == label])),
    } for label, label_name in ((0, "natural"), (1, "ai"))}
        for index, condition in enumerate(config["conditions"]) if index}


def screen_iteration(table, config, manifest_sha, old_worst_auc):
    """Identical mathematical/evaluation path for constructed controls and real cache."""
    arrays, labels = paired_matrices(table, "fit", config)
    candidates, rules = {}, {}
    for rank in config["removed_ranks"]:
        name = f"rank{rank}"
        rule, fit_diagnostics = fit_projection(arrays, labels, rank, clip=config["z_clip"],
                                              manifest_sha=manifest_sha)
        rule = threshold_for_rule(table, rule, config)
        auc_metrics, encoding = candidate_auc(table, rule, "selection", config)
        classification = classification_for_role(table, rule, "selection", config)
        candidates[name] = {"fit": fit_diagnostics, "selection_auc": auc_metrics,
                            "encoding_diagnostic_selection": encoding,
                            "selection_classification": classification,
                            "threshold_calibration_classification": classification_for_role(table, rule, "threshold", config),
                            "selection_score_drift": score_drift(table, rule, "selection", config),
                            "worst_processed_auc": min(auc_metrics[c]["auc"] for c in ("transfer", "redigital"))}
        rules[name] = rule
    for name, candidate in candidates.items():
        conditions = candidate["selection_classification"]["conditions"]
        criteria = {
            "projected_candidate": name != "rank0",
            "nonzero_rule": not candidate["fit"]["zero_rule"],
            "processed_auc_lower_bounds": all(candidate["selection_auc"][c]["auc_ci95"][0]
                                               > config["selection_auc_lower_gate"] for c in ("transfer", "redigital")),
            "gain_over_rank0": candidate["worst_processed_auc"] >= candidates["rank0"]["worst_processed_auc"]
                               + config["minimum_worst_auc_gain"],
            "gain_over_old_color": candidate["worst_processed_auc"] >= old_worst_auc + config["minimum_worst_auc_gain"],
            "both_processed_class_accuracies": all(conditions[c][key] >= config["minimum_processed_class_accuracy"]
                                                  for c in ("transfer", "redigital")
                                                  for key in ("real_accuracy_at_zero", "fake_accuracy_at_zero")),
        }
        candidate["criteria"] = {key: bool(value) for key, value in criteria.items()}
        candidate["selection_gate"] = all(candidate["criteria"].values())
    passing = [name for name, value in candidates.items() if value["selection_gate"]]
    chosen = min(passing, key=lambda name: (-candidates[name]["worst_processed_auc"], rules[name].removed_rank)) if passing else None
    return {"candidates": candidates, "chosen": chosen, "candidate_count": 2, "reference_count": 1}, rules


def previous_capture_diagnosis():
    """Descriptive use of exposed prior scores; never passed to the new fit/selector."""
    directory = WORK_DIR / "robust_statistics/chimera_first_iteration"
    receipt_path, csv_path = directory / "frozen_rule.json", directory / "frozen_rule.csv"
    manifest = DATA_ROOT / "manifests/chimera_bfree_manifest.csv"
    pins = {
        receipt_path: "fea8deea24980f8176daa6f9fdb58d883df138f5ca6bcd42b89c1bfe95da6005",
        manifest: "39da9e9eac3bd2d133c53769f9c5a158c1769039c67560c0bbfc4660d09a7d24",
    }
    for path, fingerprint in pins.items():
        if file_sha256(path) != fingerprint:
            raise ValueError("Prior capture diagnosis input changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = read_rows(manifest)
    condition_map = {"stylegan2_orig": "original", "recap_mac": "mac_iphone", "recap_monitor": "lg_blackfly"}
    expected = [ExpectedImage(r["src"], condition_map[r["condition"]],
                              Origin.AI if r["label"] == "FAKE" else Origin.NATURAL, r["filename"]) for r in rows]
    report, predictions = evaluate_cached_method(
        expected, [CachedRun(csv_path, condition_map=condition_map, expected_sha256=receipt["csv_sha256"])],
        method="prior-color-rule", threshold=receipt["algorithm"]["threshold"], score_kind="statistical_score")
    groups, labels = defaultdict(list), {}
    for item in expected:
        scene = item.source.split("/", 1)[0]
        labels[item.source] = item.label
        groups[scene, item.condition].append({"src": item.source, "label": "FAKE" if item.label == Origin.AI else "REAL",
                                             "score": predictions[item.condition][item.source].margin})
    scenes = {scene: {condition: evaluate(values, "score")
                     for (current_scene, condition), values in groups.items() if current_scene == scene}
              for scene in sorted({key[0] for key in groups})}
    shifts = {}
    for condition in ("mac_iphone", "lg_blackfly"):
        shifts[condition] = {}
        for label, name in ((Origin.NATURAL, "natural"), (Origin.AI, "ai")):
            sources = sorted(s for s, truth in labels.items() if truth == label)
            delta = np.array([predictions[condition][s].score - predictions["original"][s].score for s in sources])
            shifts[condition][name] = {"sources": len(sources), "median_score_change": float(np.median(delta)),
                                      "mean_score_change": float(delta.mean()),
                                      "positive_shift_fraction": float(np.mean(delta > 0))}
    return {"scope": "post-hoc descriptive failure diagnosis; scores already exposed; not causal identification",
            "report": report, "scene_strata": scenes, "class_score_shifts": shifts,
            "inputs": {str(p): fingerprint for p, fingerprint in pins.items()},
            "csv_sha256": receipt["csv_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-serialization", action="store_true",
                        help="Recompute after JSON serialization failure; compare all three provisional rules")
    args = parser.parse_args()
    provisional = {f"rank{rank}_rule.json" for rank in (0, 1, 3)}
    if OUTPUT.exists() and (not args.recover_serialization
                            or {p.name for p in OUTPUT.iterdir()} != provisional):
        raise FileExistsError("Second iteration exists; preserve registered results")
    started, clock_start = datetime.now(timezone.utc).isoformat(), perf_counter()
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    inventory, audit = verified_inventory()
    receipt = json.loads((FIRST / "features.json").read_text(encoding="utf-8"))
    if receipt["code_fingerprints"] != code_fingerprints():
        raise ValueError("Frozen feature code/config changed")
    if file_sha256(FIRST / "features.csv") != receipt["csv_sha256"]:
        raise ValueError("Frozen feature CSV changed")
    screening_path = FIRST / "screening.json"
    if file_sha256(screening_path) != "d900918cb74ece223a14503af5d13a9084988d0324d07352ff9384c89a2a7d49":
        raise ValueError("Prior selection receipt changed")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    old_rule = StatisticsRule.load(FIRST / "selected_rule.json")
    if file_sha256(FIRST / "selected_rule.json") != screening["selected_rule_sha256"]:
        raise ValueError("Prior color rule changed")
    table = validate_feature_rows(read_rows(FIRST / "features.csv"), inventory, settings())
    result, rules = screen_iteration(table, config, audit["development_manifest_sha256"],
                                    screening["candidates"][screening["chosen"]]["worst_processed_selection_auc"])
    result["previous_rule_rr_score_drift"] = score_drift(table, old_rule, "selection", config)
    result["previous_rule_capture_diagnosis"] = previous_capture_diagnosis()
    result["rule_files"] = {}
    for name, rule in rules.items():
        path = OUTPUT / f"{name}_rule.json"
        if path.exists() and rule.fingerprint != type(rule).load(path).fingerprint:
            raise ValueError("Provisional rule differs from validated recomputation")
        result["rule_files"][name] = {"path": str(path), "parameter_sha256": rule.fingerprint}
    result.update({"started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_seconds": perf_counter() - clock_start,
                   "scope": "second exploratory iteration; identical previously exposed development roles",
                   "config": config, "development_sources": audit["development_sources"],
                   "development_manifest_sha256": audit["development_manifest_sha256"],
                   "feature_csv_sha256": receipt["csv_sha256"],
                   "code_fingerprints": {name: file_sha256(REPO_ROOT / name) for name in (
                       "src/palimpsest/detection/algorithms/local_statistics/projection.py",
                       "experiments/origin_detection/paired_projection/rr/run_iteration.py",
                       "experiments/origin_detection/paired_projection/rr/README.md",
                       "configs/evaluation/rr_paired_projection.toml",
                       "experiments/origin_detection/robust_statistics/rr/evaluate_features.py")},
                   "old_feature_code_fingerprints": receipt["code_fingerprints"],
                   "reserved_images_opened": 0, "new_chimera_inference": False,
                   "recovered_serialization_failure": args.recover_serialization,
                   "status": "RR development gate passed; frozen diagnosis pending" if result["chosen"]
                   else "no projected candidate passed; stop this recipe without holdout or speed optimization"})
    # Serialize before writing any new artifact: a failed encoding must not leave
    # a partly signed experiment. Recovery never changes provisional rules.
    json.dumps(result, allow_nan=False)
    OUTPUT.mkdir(parents=True, exist_ok=args.recover_serialization)
    for name, rule in rules.items():
        path = OUTPUT / f"{name}_rule.json"
        if not path.exists():
            rule.save(path)
        result["rule_files"][name]["sha256"] = file_sha256(path)
    write_json(OUTPUT / "iteration.json", result)
    print(json.dumps({"chosen": result["chosen"], "status": result["status"],
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
