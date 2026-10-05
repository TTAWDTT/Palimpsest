"""Evaluate official D3 head and CLIP ViT-L/14 on RRDataset single images."""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, MODELS_ROOT
from palimpsest.detection.baselines.d3 import D3Detector, CLIP_SHA256
from palimpsest.detection.baselines import d3 as d3_backend
from palimpsest.detection.files import predict_file

from palimpsest.io.hashing import file_sha256 as sha256

from palimpsest.paths import WORK_DIR

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path


from palimpsest.evaluation.pairing import paired_change
from palimpsest.evaluation.timing import percentile
from palimpsest.evaluation.classification import evaluate


VENDOR = WORK_DIR / "vendor" / "d3"


ROOT = DATA_ROOT / "derived/rr_test"
MANIFEST = DATA_ROOT / "manifests/rr_test_files.csv"
TRAINVAL_MANIFEST = DATA_ROOT / "manifests/rr_trainval_files.csv"
CLIP_CHECKPOINT = MODELS_ROOT / "d3/ViT-L-14.pt"
HEAD_CHECKPOINT = VENDOR / "ckpt" / "classifier.pth"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_test_rows(limit: int) -> tuple[list[dict[str, str]], int]:
    test = read_csv(MANIFEST)
    trainval_hashes = {row["sha256"] for row in read_csv(TRAINVAL_MANIFEST)}
    excluded = {
        f"{row['label']}/{row['source_id']}"
        for row in test
        if row["condition"] == "original" and row["sha256"] in trainval_hashes
    }
    if len(excluded) != 14:
        raise ValueError(
            f"Expected 14 exact-overlap source groups, found {len(excluded)}"
        )
    selected = [
        row for row in test if f"{row['label']}/{row['source_id']}" not in excluded
    ]
    if limit:
        ids = {
            label: set(
                sorted(
                    {row["source_id"] for row in selected if row["label"] == label},
                    key=lambda source: hashlib.sha256(
                        f"rr-fourier-pilot-20260924/{label}/{source}".encode()
                    ).digest(),
                )[:limit]
            )
            for label in ("ai", "real")
        }
        selected = [row for row in selected if row["source_id"] in ids[row["label"]]]
    return selected, len(excluded)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    result: dict[str, object] = {}
    for condition, group in by_condition.items():
        metric_rows = [
            {"src": row["src"], "label": row["label"], "D3": row["score"]}
            for row in group
        ]
        timings = sorted(float(row["end_to_end_ms"]) for row in group)
        result[condition] = {
            "metrics": evaluate(metric_rows, "D3"),
            "timing_ms_p50": statistics.median(timings),
            "timing_ms_p95": percentile(timings, 0.95),
        }
    sources = {
        condition: {str(row["src"]): row for row in group}
        for condition, group in by_condition.items()
    }
    result["paired_changes"] = {
        condition: paired_change(sources["original"], sources[condition])
        for condition in ("transfer", "redigital")
        if condition in sources
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-sources-per-class", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch

    rows, excluded_count = select_test_rows(args.test_sources_per_class)
    if not rows:
        raise ValueError("No test rows selected")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_prefix.with_suffix(".csv")
    state_path = args.output_prefix.with_suffix(".state.pt")
    if (output_csv.exists() or state_path.exists()) and not args.resume:
        raise FileExistsError(f"Use --resume to continue {args.output_prefix}")
    if sha256(CLIP_CHECKPOINT) != CLIP_SHA256:
        raise ValueError("CLIP checkpoint SHA-256 mismatch")
    head_hash = sha256(HEAD_CHECKPOINT)
    protocol = {
        "test_manifest_sha256": sha256(MANIFEST),
        "trainval_manifest_sha256": sha256(TRAINVAL_MANIFEST),
        "clip_sha256": CLIP_SHA256,
        "head_sha256": head_hash,
        "script_sha256": sha256(Path(__file__)),
        "backend_sha256": sha256(Path(d3_backend.__file__)),
        "test_sources_per_class": args.test_sources_per_class,
    }
    checkpoint = (
        torch.load(state_path, map_location="cpu", weights_only=True)
        if state_path.exists()
        else None
    )
    if checkpoint and checkpoint["protocol"] != protocol:
        raise ValueError("RNG checkpoint protocol fingerprint mismatch")
    if checkpoint:
        if not output_csv.exists():
            raise ValueError("RNG checkpoint exists without inference CSV")
        with output_csv.open("rb") as handle:
            prefix = handle.read(int(checkpoint["csv_offset"]))
        if (
            len(prefix) != checkpoint["csv_offset"]
            or hashlib.sha256(prefix).hexdigest() != checkpoint["csv_sha256"]
        ):
            raise ValueError("Inference CSV prefix fingerprint mismatch")
        previous = list(csv.DictReader(io.StringIO(prefix.decode("utf-8"))))
    else:
        if output_csv.exists():
            raise ValueError(
                "Existing inference CSV has no corresponding RNG checkpoint"
            )
        previous = []
    if any(
        row["filename"] != expected["filename"]
        or row["src"] != f"{expected['label']}/{expected['source_id']}"
        or row["label"] != ("FAKE" if expected["label"] == "ai" else "REAL")
        for row, expected in zip(previous, rows)
    ) or len(previous) > len(rows):
        raise ValueError("Existing CSV is not a prefix of the selected manifest")
    completed = int(checkpoint["index"]) if checkpoint else 0
    if completed != len(previous):
        raise ValueError("RNG checkpoint index and CSV prefix disagree")
    detector = D3Detector(VENDOR, CLIP_CHECKPOINT, HEAD_CHECKPOINT)
    if checkpoint:
        detector.cpu_rng = checkpoint["cpu_rng"]
        detector.cuda_rng = checkpoint["cuda_rng"]
    with torch.inference_mode():
        results: list[dict[str, object]] = list(previous)
        fields = (
            "filename",
            "condition",
            "src",
            "label",
            "score",
            "decode_preprocess_ms",
            "end_to_end_ms",
        )
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(previous)
            for index, row in enumerate(rows[completed:], completed + 1):
                file_result = predict_file(detector, ROOT / row["filename"])
                score = file_result.prediction.score
                decode_ms = (
                    file_result.decode_ms
                    + file_result.prediction.timing_ms["preprocess"]
                )
                total_ms = file_result.end_to_end_ms
                if not math.isfinite(score):
                    raise ValueError(f"Nonfinite D3 score at {row['filename']}")
                result = {
                    "filename": row["filename"],
                    "condition": row["condition"],
                    "src": f"{row['label']}/{row['source_id']}",
                    "label": "FAKE" if row["label"] == "ai" else "REAL",
                    "score": score,
                    "decode_preprocess_ms": decode_ms,
                    "end_to_end_ms": total_ms,
                }
                writer.writerow(result)
                results.append(result)
                if index % 1000 == 0 or index == len(rows):
                    handle.flush()
                    os.fsync(handle.fileno())
                    csv_offset = output_csv.stat().st_size
                    temporary = state_path.with_suffix(".state.tmp")
                    torch.save(
                        {
                            "index": index,
                            "cpu_rng": detector.cpu_rng,
                            "cuda_rng": detector.cuda_rng,
                            "protocol": protocol,
                            "csv_offset": csv_offset,
                            "csv_sha256": sha256(output_csv),
                        },
                        temporary,
                    )
                    temporary.replace(state_path)
                    print(f"test {index}/{len(rows)}", flush=True)
    report = {
        "method": "D3 CVPR2025 official CLIP ViT-L/14 + classifier head",
        "vendor_commit": "14f21ad3797ef1e42f2f6090aa8ad4fabf07896c",
        "clip_sha256": CLIP_SHA256,
        "head_sha256": head_hash,
        "global_seed": 418,
        "decision_threshold": "raw logit > 0 is AI",
        "excluded_exact_overlap_sources": excluded_count,
        "test_sources_per_class": args.test_sources_per_class,
        "test": summarize(results),
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"output": str(args.output_prefix), "test_rows": len(results)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
