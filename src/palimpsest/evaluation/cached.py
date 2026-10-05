"""Evaluate existing CSV scores against an independent expected image inventory.

No inference, fitting, image decoding or normalized intermediate CSV is needed.
Dataset adapters supply explicit condition mappings and original-content labels.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from palimpsest.contracts import Origin, Prediction
from palimpsest.io.hashing import file_sha256
from .detection import DetectionObservation, evaluate_detection
from .timing import summarize_timing, percentile


@dataclass(frozen=True)
class ExpectedImage:
    source: str
    condition: str
    label: Origin
    filename: str


@dataclass(frozen=True)
class CachedRun:
    path: Path
    condition_map: Mapping[str, str] = field(default_factory=dict)
    fixed_condition: str | None = None
    input_conditions: tuple[str, ...] | None = None
    expected_sha256: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)


def read_cached_run(
    run: CachedRun,
    expected: Mapping[tuple[str, str], ExpectedImage],
    *,
    method: str,
    threshold: float,
    score_kind: str,
    excluded_sources: set[str],
    selected_sources: set[str] | None = None,
) -> tuple[dict[tuple[str, str], tuple[Prediction, dict]], dict]:
    fingerprint = file_sha256(run.path)
    if run.expected_sha256 is not None and fingerprint != run.expected_sha256:
        raise ValueError(f"CSV fingerprint mismatch: {run.path}")
    if bool(run.condition_map) == (run.fixed_condition is not None):
        raise ValueError("Use exactly one of condition_map and fixed_condition")
    observations, filenames = {}, set()
    total, skipped = 0, 0
    with run.path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            total += 1
            filename, source = row["filename"], row["src"]
            if not filename or filename in filenames or not source:
                raise ValueError(f"Empty or duplicate filename/source: {filename}")
            filenames.add(filename)
            if row.get("error"):
                raise ValueError(f"Inference error: {filename}: {row['error']}")
            label = {"REAL": Origin.NATURAL, "FAKE": Origin.AI}[row["label"]]
            timings = {
                key: float(row[key])
                for key in (
                    "end_to_end_ms",
                    "decode_preprocess_ms",
                    "gpu_transfer_forward_ms",
                )
                if key in row
            }
            if "end_to_end_ms" not in timings:
                raise ValueError(f"Missing end-to-end timing: {filename}")
            prediction = Prediction(
                method, float(row["score"]), threshold, score_kind, timing_ms=timings
            )
            original_condition = filename.split("/", 1)[0]
            if run.condition_map and original_condition not in (
                run.input_conditions or run.condition_map
            ):
                raise ValueError(f"Unexpected input condition: {filename}")
            if "condition" in row and row["condition"] != original_condition:
                raise ValueError(f"Condition differs from filename: {filename}")
            condition = run.fixed_condition or run.condition_map.get(original_condition)
            if (
                condition is None
                or source in excluded_sources
                or (selected_sources is not None and source not in selected_sources)
            ):
                skipped += 1
                continue
            key = condition, source
            if key in observations:
                raise ValueError(f"Duplicate source/condition: {key}")
            if key not in expected:
                raise ValueError(f"Unexpected source/condition: {key}")
            item = expected[key]
            if item.filename != filename or item.label != label:
                raise ValueError(f"Prediction differs from expected inventory: {key}")
            observations[key] = prediction, row
    return observations, {
        "path": str(run.path),
        "sha256": fingerprint,
        "csv_rows": total,
        "included_rows": len(observations),
        "skipped_rows": skipped,
        "condition_map": dict(run.condition_map),
        "fixed_condition": run.fixed_condition,
        "provenance": dict(run.provenance),
    }


def evaluate_cached_method(
    expected_images: list[ExpectedImage],
    runs: list[CachedRun],
    *,
    method: str,
    threshold: float = 0.0,
    score_kind: str = "logit",
    excluded_sources: set[str] | None = None,
    selected_sources: set[str] | None = None,
    reference: str = "original",
) -> tuple[dict, dict[str, dict[str, Prediction]]]:
    expected = {(item.condition, item.source): item for item in expected_images}
    if not expected or len(expected) != len(expected_images):
        raise ValueError(
            "Expected inventory must be nonempty with unique source/condition"
        )
    if any(not i.filename or not i.source or not i.condition for i in expected_images):
        raise ValueError("Expected inventory contains empty identifiers")
    observations, provenance = {}, []
    for run in runs:
        rows, info = read_cached_run(
            run,
            expected,
            method=method,
            threshold=threshold,
            score_kind=score_kind,
            excluded_sources=excluded_sources or set(),
            selected_sources=selected_sources,
        )
        if observations.keys() & rows.keys():
            raise ValueError("Multiple runs claim the same source/condition")
        observations.update(rows)
        provenance.append(info)
    if observations.keys() != expected.keys():
        raise ValueError(
            f"Coverage mismatch: missing={len(expected.keys() - observations.keys())}, "
            f"extra={len(observations.keys() - expected.keys())}"
        )
    result = evaluate_detection(
        [
            DetectionObservation(
                source, condition, expected[condition, source].label, value[0]
            )
            for (condition, source), value in observations.items()
        ],
        reference=reference,
    )
    predictions, timings = {}, {}
    for (condition, source), (prediction, row) in observations.items():
        predictions.setdefault(condition, {})[source] = prediction
        timings.setdefault(condition, []).append(row)
    result["input_runs"] = provenance
    result["score_kind"] = score_kind
    result["timing_scope"] = (
        "recorded single-image decode/preprocess/forward; excludes model startup"
    )
    result["timing_ms"] = {}
    for condition, rows in {
        **timings,
        "all": [v[1] for v in observations.values()],
    }.items():
        endpoints = [float(row["end_to_end_ms"]) for row in rows]
        summary = {
            "end_to_end_ms": {
                "p50": percentile(endpoints, 0.5),
                "p95": percentile(endpoints, 0.95),
            }
        }
        if all(
            "gpu_transfer_forward_ms" in row and "decode_preprocess_ms" in row
            for row in rows
        ):
            summary = summarize_timing(rows)
        result["timing_ms"][condition] = summary
    return result, predictions


def classification_table(methods: Mapping[str, dict]) -> str:
    """Render the same columns for full datasets and propagation controls."""
    lines = [
        "| 方法 | 条件 | 图像数 | AUC | BA | 真实类正确率 | AI类正确率 | 配对BA变化（百分点） | E2E P50/P95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in methods.items():
        for condition, metrics in report["conditions"].items():
            change = report["paired_changes"].get(condition)
            delta = (
                f"{100 * change['balanced_accuracy_change']:+.2f}" if change else "—"
            )
            timing = report["timing_ms"][condition]["end_to_end_ms"]
            numbers = [
                metrics[key] * 100
                for key in (
                    "auc",
                    "balanced_accuracy_at_zero",
                    "real_accuracy_at_zero",
                    "fake_accuracy_at_zero",
                )
            ]
            lines.append(
                f"| {name} | {condition} | {metrics['images']} | "
                + " | ".join(f"{v:.2f}%" for v in numbers)
                + f" | {delta} | {timing['p50']:.1f}/{timing['p95']:.1f} |"
            )
    return "\n".join(lines)
