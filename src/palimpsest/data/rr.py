"""RR released-manifest pairing and historical coarse RGB decoding.

Excludes known exact train/val overlaps, not unexamined near duplicates.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from palimpsest.paths import DATA_ROOT, WORK_DIR

ROOT = DATA_ROOT / "derived/rr_test"
MANIFEST = DATA_ROOT / "manifests/rr_test_files.csv"
OVERLAP_AUDIT = WORK_DIR / "rr_bfree_evaluation.json"


def load_pairs(
    manifest: Path = MANIFEST,
    overlap_audit: Path = OVERLAP_AUDIT,
    *,
    expected_sources: int = 16986,
    expected_redigital: int = 16985,
) -> dict[str, dict[str, dict[str, str]]]:
    excluded = set(
        json.loads(overlap_audit.read_text(encoding="utf-8"))["trainval_overlap_audit"][
            "excluded_source_ids"
        ]
    )
    sources: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = f"{row['label']}/{row['source_id']}"
            if source in excluded:
                continue
            condition = row["condition"]
            if condition in sources[source]:
                raise ValueError(f"duplicate {source} {condition}")
            sources[source][condition] = row
    if len(sources) != expected_sources:
        raise ValueError(f"unexpected source count: {len(sources)}")
    if sum("redigital" in rows for rows in sources.values()) != expected_redigital:
        raise ValueError("unexpected redigital pair count")
    return sources


def load_small_image(
    row: dict[str, str], side: int, *, root: Path = ROOT
) -> tuple[np.ndarray, str | None]:
    with Image.open(root / row["filename"]) as opened:
        jpeg_table = None
        if opened.format == "JPEG" and opened.quantization:
            table = opened.quantization.get(0)
            if table:
                jpeg_table = hashlib.sha256(
                    np.asarray(table, dtype=np.uint16).tobytes()
                ).hexdigest()[:16]
        image = ImageOps.exif_transpose(opened).convert("RGB")
        array = (
            np.asarray(
                image.resize((side, side), Image.Resampling.BICUBIC), dtype=np.float32
            )
            / 255
        )
    return array, jpeg_table
