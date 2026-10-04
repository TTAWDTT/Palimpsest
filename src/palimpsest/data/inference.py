"""Validate ordered single-image predictions against their signed manifest.

This checks identity, coverage and scores; it does not assert training-set
independence or attest to the physical process that produced an image.
"""

import json
import math
from pathlib import Path

from palimpsest.io.tables import read_rows as rows


def validated_inference(
    manifest_path: Path, csv_path: Path, summary_path: Path, expected_count: int
) -> dict[str, list[dict[str, str]]]:
    manifest = rows(manifest_path)
    inference = rows(csv_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(manifest) != expected_count or len(inference) != expected_count:
        raise RuntimeError(f"manifest/inference rows not {expected_count}")
    if (
        summary["completed_images"] != expected_count
        or summary["remaining_images"] != 0
        or summary["errors"] != 0
        or summary["stopping_error"]
    ):
        raise RuntimeError("inference summary incomplete")
    grouped = {}
    seen = set()
    for expected, row in zip(manifest, inference):
        if any(row[field] != expected[field] for field in ("filename", "src", "label")):
            raise RuntimeError(f"identity/order mismatch for {expected['filename']}")
        if (
            row["filename"] in seen
            or row["error"]
            or not math.isfinite(float(row["score"]))
        ):
            raise RuntimeError(
                f"duplicate, error or nonfinite score: {row['filename']}"
            )
        seen.add(row["filename"])
        grouped.setdefault(expected["condition"], []).append(row)
    return grouped
