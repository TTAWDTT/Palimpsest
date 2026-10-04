"""Report ReWIND batch-one B-Free timings by patch-projection execution mode."""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

from palimpsest.evaluation.timing import percentile

import csv
import json
import math
import statistics


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    result: dict[str, object] = {"images": len(rows)}
    for key in ("decode_preprocess_ms", "gpu_transfer_forward_ms", "end_to_end_ms"):
        values = [float(row[key]) for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Invalid timing in {key}")
        result[key] = {
            "p50": statistics.median(values),
            "p95": percentile(values, 0.95),
        }
    return result


def main() -> None:
    input_path = WORK_DIR / "bfree_rewind_complete.csv"
    output_path = WORK_DIR / "bfree_rewind_latency_by_mode.json"
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5582 or any(row["error"] for row in rows):
        raise ValueError("Expected all 5,582 successful images")
    if any(not math.isfinite(float(row["score"])) for row in rows):
        raise ValueError("Nonfinite detector score")
    result = {
        "untiled_prefix": summarize(rows[:2159]),
        "tiled_suffix": summarize(rows[2159:]),
        "all_mixed_modes": summarize(rows),
        "note": "Batch=1, prewarmed per pass; tiled suffix uses 64-patch projection tiles.",
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
