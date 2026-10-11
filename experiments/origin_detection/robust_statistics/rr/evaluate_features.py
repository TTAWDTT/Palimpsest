"""Screen registered feature rules; never turn development into a blind test."""

from dataclasses import replace
import json

import numpy as np

from palimpsest.contracts import Origin, Prediction
from palimpsest.detection.algorithms.local_statistics.detector import StatisticsRule
from palimpsest.detection.algorithms.local_statistics.features import FAMILIES, FEATURE_NAMES
from palimpsest.evaluation.classification import auc
from palimpsest.evaluation.detection import DetectionObservation, evaluate_detection
from palimpsest.evaluation.cached import CachedRun, ExpectedImage, evaluate_cached_method
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.baselines.evaluate_rr_suite import FILES
from .protocol import RESULT, read_csv, settings, verified_inventory, write_json
from .run_features import code_fingerprints


def validate_feature_rows(rows, inventory, config):
    expected = {row["filename"]: row for row in inventory}
    keys, output = set(), {}
    if len(expected) != len(inventory) or not inventory:
        raise ValueError("Invalid independent inventory")
    for row in rows:
        name = row["filename"]
        edge = int(row["long_edge"])
        variant = row["variant"]
        if name not in expected or edge not in config["long_edges"] or variant not in ("raw", "jpeg90_444_after_resize"):
            raise ValueError("Unexpected feature identity")
        original = expected[name]
        if (row["src"] != original["source_group"] or row["condition"] != original["condition"]
                or row["role"] != original["algorithm_role"]
                or row["label"] != ("FAKE" if original["label"] == "ai" else "REAL")):
            raise ValueError("Feature identity/label/role differs from inventory")
        key = name, edge, variant
        if key in keys:
            raise ValueError("Duplicate feature record")
        keys.add(key)
        values = np.array([float(row[f]) for f in FEATURE_NAMES])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite feature record")
        output.setdefault((edge, variant, row["role"], row["condition"]), []).append((row, values))
    expected_keys = {(name, edge, variant) for name in expected for edge in config["long_edges"]
                     for variant in ("raw", "jpeg90_444_after_resize")}
    if keys != expected_keys:
        raise ValueError("Feature coverage mismatch")
    return output


def matrix(table, edge, variant, role, condition):
    values = sorted(table[edge, variant, role, condition], key=lambda item: item[0]["src"])
    return ([v[0] for v in values], np.array([v[1] for v in values]),
            np.array([v[0]["label"] == "FAKE" for v in values], dtype=int))


def fit_family(table, family, edge, config, manifest_sha):
    _, transfer, labels = matrix(table, edge, "raw", "fit", "transfer")
    _, redigital, other_labels = matrix(table, edge, "raw", "fit", "redigital")
    if not np.array_equal(labels, other_labels):
        raise ValueError("Fit sources are misaligned")
    pool = np.concatenate([transfer, redigital])
    pool_labels = np.tile(labels, 2)
    chosen, directions, diagnostics = [], [], []
    for index in FAMILIES[family]:
        pool_auc = auc(list(zip(pool[:, index], pool_labels)))
        direction = 1 if pool_auc >= 0.5 else -1
        condition_aucs = [auc(list(zip(direction * values[:, index], labels)))
                          for values in (transfer, redigital)]
        kept = bool(min(condition_aucs) >= config["minimum_fit_auc"])
        if kept:
            chosen.append(index)
            directions.append(direction)
        diagnostics.append({"feature": FEATURE_NAMES[index], "direction": direction,
                            "transfer_auc": condition_aucs[0], "redigital_auc": condition_aucs[1],
                            "kept": kept})
    original = matrix(table, edge, "raw", "fit", "original")[1]
    calibration = np.concatenate([original, transfer, redigital])[:, chosen]
    center = np.median(calibration, axis=0)
    mad = np.maximum(np.median(np.abs(calibration - center), axis=0), 1e-8)
    rule = StatisticsRule(family, edge, tuple(chosen), tuple(center), tuple(mad),
                          tuple(directions), clip=config["z_clip"], fit_manifest_sha256=manifest_sha)
    return rule, diagnostics


def auc_intervals(scores, labels, config):
    # Same source draws are used for every condition. This preserves pairing.
    indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    if not all(len(v) for v in indices):
        raise ValueError("Bootstrap requires both source classes")
    rng = np.random.default_rng(config["seed"])
    output = {c: [] for c in scores}
    for _ in range(config["bootstrap_repetitions"]):
        draw = np.concatenate([rng.choice(v, len(v), replace=True) for v in indices])
        for condition, values in scores.items():
            output[condition].append(auc(list(zip(values[draw], labels[draw]))))
    return {condition: np.quantile(values, [0.025, 0.975]).tolist()
            for condition, values in output.items()}


def candidate_auc(table, rule, role, config):
    raw_scores, labels, source_names, conditions = {}, None, None, {}
    for condition in config["conditions"]:
        rows, values, current_labels = matrix(table, rule.long_edge, "raw", role, condition)
        names = [row["src"] for row in rows]
        if source_names is not None and names != source_names:
            raise ValueError("Condition source alignment differs")
        labels, source_names = current_labels, names
        raw_scores[condition] = rule.score(values)
        conditions[condition] = {"auc": auc(list(zip(raw_scores[condition], labels)))}
    intervals = auc_intervals({c: raw_scores[c] for c in ("transfer", "redigital")}, labels, config)
    for condition, interval in intervals.items():
        conditions[condition]["auc_ci95"] = interval
    diagnostic = {}
    for condition in config["conditions"]:
        _, values, current_labels = matrix(table, rule.long_edge, "jpeg90_444_after_resize", role, condition)
        diagnostic[condition] = {"auc": auc(list(zip(rule.score(values), current_labels)))}
    return conditions, diagnostic


def threshold_for_rule(table, rule, config):
    observations, scores = {}, []
    for condition in ("transfer", "redigital"):
        rows, values, labels = matrix(table, rule.long_edge, "raw", "threshold", condition)
        observations[condition] = rule.score(values), labels
        scores.extend(rule.score(values).tolist())
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), unique]
    ranked = []
    for threshold in candidates:
        metrics = []
        for values, labels in observations.values():
            correctness = (values > threshold) == labels.astype(bool)
            metrics.append(float((correctness[labels == 0].mean() + correctness[labels == 1].mean()) / 2))
        ranked.append((min(metrics), -abs(float(threshold)), -float(threshold), float(threshold)))
    return replace(rule, threshold=max(ranked)[-1])


def selection_null_control(scores, labels, config):
    """Group-label permutation preserves paired versions and all six searches."""
    def best_worst(values):
        return max(min(auc(list(zip(condition, values))) for condition in candidate.values())
                   for candidate in scores.values())
    observed = best_worst(labels)
    rng = np.random.default_rng(config["seed"] + 1)
    null = [best_worst(rng.permutation(labels)) for _ in range(config["bootstrap_repetitions"])]
    return {"observed_max_worst_auc": observed,
            "null_max_worst_auc_ci95": np.quantile(null, [0.025, 0.975]).tolist(),
            "permutation_p": (1 + sum(value >= observed for value in null)) / (1 + len(null)),
            "permutations": len(null),
            "scope": "fit rules fixed; selection label permutation accounts for six-candidate search, "
                     "does not remove dataset/encoding bias or create an independent test"}


def classification_for_role(table, rule, role, config):
    observations = []
    for condition in config["conditions"]:
        rows, values, labels = matrix(table, rule.long_edge, "raw", role, condition)
        for row, score, label in zip(rows, rule.score(values), labels):
            observations.append(DetectionObservation(
                row["src"], condition, Origin.AI if label else Origin.NATURAL,
                Prediction(f"{rule.family}-{rule.long_edge}", float(score), rule.threshold, "statistical_score")))
    return evaluate_detection(observations)


def main():
    if (RESULT / "screening.json").exists():
        raise FileExistsError("Screening exists; preserve the original selection record")
    inventory, audit = verified_inventory()
    receipt = json.loads((RESULT / "features.json").read_text(encoding="utf-8"))
    if receipt["code_fingerprints"] != code_fingerprints():
        raise ValueError("Feature code/config changed after extraction")
    if receipt["csv_sha256"] != file_sha256(RESULT / "features.csv"):
        raise ValueError("Feature CSV fingerprint changed")
    config = settings()
    table = validate_feature_rows(read_csv(RESULT / "features.csv"), inventory, config)
    rules, candidates, null_scores = {}, {}, {}
    for edge in config["long_edges"]:
        for family in config["families"]:
            name = f"{family}-{edge}"
            rule, diagnostics = fit_family(table, family, edge, config, audit["development_manifest_sha256"])
            metrics, encoding = candidate_auc(table, rule, "selection", config)
            gate = bool(rule.indices) and all(metrics[c]["auc_ci95"][0] > config["selection_auc_lower_gate"]
                                             for c in ("transfer", "redigital"))
            candidates[name] = {"features": diagnostics, "retained": len(rule.indices),
                                "selection": metrics, "encoding_diagnostic_selection": encoding,
                                "selection_gate": gate,
                                "worst_processed_selection_auc": min(metrics[c]["auc"] for c in ("transfer", "redigital"))}
            rules[name] = rule
            null_scores[name] = {}
            for condition in ("transfer", "redigital"):
                _, values, null_labels = matrix(table, edge, "raw", "selection", condition)
                null_scores[name][condition] = rule.score(values)
            print(name, candidates[name]["retained"], metrics, "gate", gate, flush=True)
    passing = [name for name, values in candidates.items() if values["selection_gate"]]
    chosen = sorted(passing, key=lambda name: (-candidates[name]["worst_processed_selection_auc"], name))[0] if passing else None
    result = {"scope": "development screening, not final evidence or independent test",
              "candidates": candidates, "chosen": chosen, "candidate_count": len(candidates),
              "development_manifest_sha256": audit["development_manifest_sha256"],
              "feature_csv_sha256": receipt["csv_sha256"],
              "analysis_code_sha256": file_sha256(REPO_ROOT / "experiments/origin_detection/robust_statistics/rr/evaluate_features.py"),
              "gate_limit": "six exploratory selections; pointwise intervals are not confirmatory/multiplicity-corrected",
              "final_internal_holdout_run": False}
    result["selection_null_control"] = selection_null_control(null_scores, null_labels, config)
    selection = [row for row in inventory if row["algorithm_role"] == "selection"]
    expected = [ExpectedImage(row["source_group"], row["condition"],
                              Origin.AI if row["label"] == "ai" else Origin.NATURAL,
                              row["filename"]) for row in selection]
    result["cached_baseline_selection"] = {}
    for method, (filename, fingerprint) in FILES.items():
        report, _ = evaluate_cached_method(
            expected, [CachedRun(WORK_DIR / filename,
                                condition_map={c: c for c in config["conditions"]},
                                expected_sha256=fingerprint)], method=method,
            score_kind="margin" if method == "Benford-RF" else "logit",
            selected_sources={row["source_group"] for row in selection})
        result["cached_baseline_selection"][method] = report
    if chosen:
        rule = threshold_for_rule(table, rules[chosen], config)
        rule_path = RESULT / "selected_rule.json"
        if rule_path.exists():
            if StatisticsRule.load(rule_path).fingerprint != rule.fingerprint:
                raise ValueError("Existing provisional rule differs; do not overwrite it")
        else:
            rule.save(rule_path)
        result["selected_rule_sha256"] = file_sha256(RESULT / "selected_rule.json")
        result["selected_threshold"] = rule.threshold
        result["selected_selection_classification"] = classification_for_role(table, rule, "selection", config)
        result["threshold_calibration_classification"] = classification_for_role(table, rule, "threshold", config)
        result["selected_rule_indices"] = list(rule.indices)
        result["status"] = "provisional candidate; encoding/content bias and fresh validation still required"
    else:
        result["status"] = "no candidate passed; stop this feature recipe, do not tune on holdout"
    write_json(RESULT / "screening.json", result)


if __name__ == "__main__":
    main()
