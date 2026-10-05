"""Freeze the small verified Dragotti Range probe as one machine-readable record."""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import REPO_ROOT

import hashlib
import json
from pathlib import Path


ROOT = REPO_ROOT
DATA = DATA_ROOT / "derived/dragotti_probe"
OUTPUT = ROOT / "outputs" / "03_过程模拟" / "Dragotti跨设备配对审计_2026-09-24.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_source(path: Path, expected_sha256: str):
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"source hash mismatch: {path}")
    return {"file": str(path), "bytes": len(content), "sha256": digest}


def collect(key: str, source: dict, recaptured_manifest: Path, analysis: Path):
    manifest = load(recaptured_manifest)
    measurements = load(analysis)
    if manifest["source_key"] != key or len(manifest["records"]) != len(
        measurements["records"]
    ):
        raise RuntimeError(f"record mismatch for {key}")
    member_by_path = {row["file"]: row for row in manifest["records"]}
    merged = []
    for row in measurements["records"]:
        member = member_by_path.pop(row["file"])
        content = Path(row["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != member["sha256"]:
            raise RuntimeError(f"recapture hash mismatch: {row['file']}")
        merged.append({**member, **row})
    if member_by_path:
        raise RuntimeError(f"missing analyses for {key}: {member_by_path}")
    return {
        "source_key": key,
        "original": source,
        "recapture_archive_url": manifest["archive_url"],
        "recapture_archive_bytes_from_HTTP_Range": manifest["archive_bytes"],
        "recaptured": merged,
    }


def main():
    source_015 = checked_source(
        DATA / "original_D40_015.jpg",
        "c6b389d3486ee469b9eb6879837287f5e996b32060f5018c786cb27b591ae41b",
    )
    original_016 = load(DATA / "D40-016_original_manifest.json")
    source_016 = checked_source(
        Path(original_016["records"][0]["file"]), original_016["records"][0]["sha256"]
    )
    bundle = {
        "date": "2026-09-24",
        "status": "selected official ZIP members only; full ZIP hashes not audited",
        "original_archive_url": original_016["archive_url"],
        "original_archive_bytes_from_HTTP_Range": original_016["archive_bytes"],
        "registration": {
            "method": "SIFT + 0.75 ratio + RANSAC homography",
            "longest_edge_for_features": 1200,
            "ransac_threshold_px": 3,
            "conditional_pitch_formula": "center vertical homography scale * source_height / 1080",
            "conditional_pitch_assumptions": [
                "source aspect ratio preserved and fills 1080 display rows",
                "published recapture only cropped, never resized",
            ],
        },
        "samples": [
            collect(
                "D40-015",
                source_015,
                DATA / "D40-015_manifest.json",
                WORK_DIR / "dragotti_D40_015_analysis.json",
            ),
            collect(
                "D40-016",
                source_016,
                DATA / "D40-016_recaptured_manifest.json",
                WORK_DIR / "dragotti_D40_016_analysis.json",
            ),
        ],
    }
    OUTPUT.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {OUTPUT} with {sum(len(x['recaptured']) for x in bundle['samples'])} recaptures"
    )


if __name__ == "__main__":
    main()
