"""Evaluate source-paired B-Free performance on verified Chimera recaptures."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST_SHA = "39da9e9eac3bd2d133c53769f9c5a158c1769039c67560c0bbfc4660d09a7d24"
CONDITIONS = ("stylegan2_orig", "recap_mac", "recap_monitor")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def summary(values: np.ndarray) -> dict[str, float]:
    return {"p50": float(np.quantile(values, .5)),
            "p95": float(np.quantile(values, .95)),
            "mean": float(values.mean())}


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for exact score ties."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both labels")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    average_ranks = first + (counts + 1) / 2
    ranks = np.repeat(average_ranks, counts)
    positive_rank_sum = float(ranks[labels[order]].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def condition_metrics(group: list[dict[str, str]]) -> dict:
    labels = np.array([row["label"] == "FAKE" for row in group], dtype=bool)
    scores = np.array([float(row["score"]) for row in group], dtype=float)
    correct = (scores > 0) == labels
    by_label = {}
    for name, truth in (("REAL", False), ("FAKE", True)):
        values = correct[labels == truth]
        by_label[name] = {"count": int(values.size), "accuracy": float(values.mean())}
    return {
        "count": len(group), "auc": float(roc_auc_score(labels, scores)),
        "zero_threshold_balanced_accuracy": float(
            (by_label["REAL"]["accuracy"] + by_label["FAKE"]["accuracy"]) / 2),
        "by_label": by_label,
        "latency_ms": {key: summary(np.array([float(row[key]) for row in group]))
                       for key in ("decode_preprocess_ms", "gpu_transfer_forward_ms", "end_to_end_ms")},
    }


def paired_metrics(original: dict[str, dict], processed: dict[str, dict],
                   rng: np.random.Generator) -> dict:
    source_ids = sorted(original)
    if set(source_ids) != set(processed):
        raise RuntimeError("paired source coverage differs")
    labels = np.array([original[src]["label"] == "FAKE" for src in source_ids], dtype=bool)
    before = np.array([float(original[src]["score"]) for src in source_ids])
    after = np.array([float(processed[src]["score"]) for src in source_ids])
    before_correct = (before > 0) == labels
    after_correct = (after > 0) == labels
    changes = after - before
    scene_strata = [np.flatnonzero(np.array([src.startswith(f"{scene}/") for src in source_ids]) &
                                    (labels == truth))
                    for scene in ("cat", "church", "horse") for truth in (False, True)]
    stratum_sizes = {len(indices) for indices in scene_strata}
    if len(stratum_sizes) != 1 or 0 in stratum_sizes:
        raise RuntimeError("expected six nonempty balanced scene/label strata")
    bootstrap = []
    bootstrap_auc = []
    for _ in range(2000):
        indices = np.concatenate([rng.choice(stratum, len(stratum), replace=True)
                                  for stratum in scene_strata])
        bootstrap.append(float((after_correct[indices].astype(np.int8) -
                                before_correct[indices].astype(np.int8)).mean()))
        bootstrap_auc.append(roc_auc_score(labels[indices], after[indices]) -
                             roc_auc_score(labels[indices], before[indices]))
    bootstrap = np.array(bootstrap)
    bootstrap_auc = np.array(bootstrap_auc)
    return {
        "paired_source_count": len(source_ids),
        "balanced_accuracy_change_pp": float(100 * (after_correct.mean() - before_correct.mean())),
        "paired_bootstrap_95ci_change_pp": (100 * np.quantile(bootstrap, [.025, .975])).tolist(),
        "auc_change": roc_auc_score(labels, after) - roc_auc_score(labels, before),
        "paired_bootstrap_95ci_auc_change": np.quantile(bootstrap_auc, [.025, .975]).tolist(),
        "score_change_mean": float(changes.mean()),
        "score_change_absolute_mean": float(np.abs(changes).mean()),
        "score_change_pearson": float(np.corrcoef(before, after)[0, 1]),
        "prediction_flip_count": int(((before > 0) != (after > 0)).sum()),
        "original_correct_to_processed_wrong": int((before_correct & ~after_correct).sum()),
        "original_wrong_to_processed_correct": int((~before_correct & after_correct).sum()),
        "by_label": {
            name: {
                "count": int(mask.sum()),
                "accuracy_change_pp": float(100 * (after_correct[mask].mean() - before_correct[mask].mean())),
                "correct_to_wrong": int((before_correct[mask] & ~after_correct[mask]).sum()),
                "wrong_to_correct": int((~before_correct[mask] & after_correct[mask]).sum()),
            } for name, mask in (("REAL", ~labels), ("FAKE", labels))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-csv", type=Path, default=Path("work/chimera_bfree_full.csv"))
    parser.add_argument("--inference-summary", type=Path, default=Path("work/chimera_bfree_full.json"))
    parser.add_argument("--manifest", type=Path, default=Path("E:/ai_image_origin_research/data/manifests/chimera_bfree_manifest.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("work/chimera_bfree_evaluation.json"))
    args = parser.parse_args()
    if file_sha256(args.manifest) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Chimera inference manifest changed")
    manifest = rows(args.manifest)
    inference = rows(args.inference_csv)
    inference_summary = json.loads(args.inference_summary.read_text(encoding="utf-8"))
    if len(manifest) != 3600 or len(inference) != 3600:
        raise RuntimeError("expected complete 3600-image manifest and inference")
    if (inference_summary["completed_images"] != 3600 or inference_summary["errors"] != 0
            or inference_summary["remaining_images"] != 0 or inference_summary["stopping_error"]):
        raise RuntimeError("inference summary not complete")
    seen = set()
    grouped = {condition: [] for condition in CONDITIONS}
    for expected, row in zip(manifest, inference):
        for field in ("filename", "src", "label"):
            if row[field] != expected[field]:
                raise RuntimeError(f"inference order or identity mismatch: {field}")
        if row["filename"] in seen:
            raise RuntimeError(f"duplicate inference filename: {row['filename']}")
        seen.add(row["filename"])
        if row["error"] or not math.isfinite(float(row["score"])):
            raise RuntimeError(f"missing or nonfinite score: {row['filename']}")
        condition = expected["condition"]
        grouped[condition].append(row)
    if any(len(group) != 1200 for group in grouped.values()):
        raise RuntimeError("condition counts differ from expected 1200")
    indexed = {condition: {row["src"]: row for row in group}
               for condition, group in grouped.items()}
    rng = np.random.default_rng(20260925)
    evaluation = {
        "scope": "Chimera Zenodo 14736478; 1200 source groups, each original and two physical screen recaptures",
        "publication": "https://www.usenix.org/system/files/usenixsecurity25-park.pdf",
        "archive_audit": "work/chimera_archive_audit.json",
        "extraction_audit": "work/chimera_extracted_audit.json",
        "model": "official B-Free BFREE_dino2reg4; zero score predicts fake",
        "inference_csv_sha256": file_sha256(args.inference_csv),
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "conditions": {condition: condition_metrics(group) for condition, group in grouped.items()},
        "paired_to_original": {
            condition: paired_metrics(indexed["stylegan2_orig"], indexed[condition], rng)
            for condition in CONDITIONS[1:]
        },
        "by_scene_class": {
            klass: {condition: condition_metrics([row for row in group if row["src"].startswith(f"{klass}/")])
                    for condition, group in grouped.items()}
            for klass in ("cat", "church", "horse")
        },
    }
    args.output_json.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"conditions": {key: {"auc": value["auc"], "ba": value["zero_threshold_balanced_accuracy"]}
                                     for key, value in evaluation["conditions"].items()},
                      "paired": {key: {"delta_ba_pp": value["balanced_accuracy_change_pp"],
                                       "flips": value["prediction_flip_count"]}
                                 for key, value in evaluation["paired_to_original"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
