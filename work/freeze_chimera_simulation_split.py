"""Freeze source-level, stratified Chimera splits before simulator calibration."""

import csv
import hashlib
import json
from pathlib import Path


SOURCE_MANIFEST = Path("E:/ai_image_origin_research/data/manifests/chimera_bfree_manifest.csv")
EXPECTED_SHA = "39da9e9eac3bd2d133c53769f9c5a158c1769039c67560c0bbfc4660d09a7d24"
OUTPUT = Path("E:/ai_image_origin_research/data/manifests/chimera_simulation_source_split.csv")
AUDIT = Path("work/chimera_simulation_source_split.json")
SALT = b"chimera-screen-physical-simulation-split-v1"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if file_hash(SOURCE_MANIFEST) != EXPECTED_SHA:
        raise RuntimeError("source manifest changed")
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 3600:
        raise RuntimeError("expected 3600 image rows")
    sources = {}
    for row in rows:
        key = row["src"]
        if key not in sources:
            scene, truth, _ = key.split("/")
            sources[key] = {"src": key, "scene": scene, "truth": truth, "conditions": set()}
        if row["label"] != ("REAL" if sources[key]["truth"] == "0_real" else "FAKE"):
            raise RuntimeError(f"label mismatch: {key}")
        sources[key]["conditions"].add(row["condition"])
    if len(sources) != 1200 or any(len(item["conditions"]) != 3 for item in sources.values()):
        raise RuntimeError("source pairing incomplete")
    groups = {}
    for item in sources.values():
        groups.setdefault((item["scene"], item["truth"]), []).append(item["src"])
    if len(groups) != 6 or any(len(values) != 200 for values in groups.values()):
        raise RuntimeError("unexpected scene/label balance")
    assignment = {}
    for group, values in groups.items():
        ordered = sorted(values, key=lambda value: hashlib.sha256(SALT + value.encode()).digest())
        for index, source in enumerate(ordered):
            assignment[source] = "calibration" if index < 40 else "development" if index < 60 else "reserved"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("src", "scene", "label", "split"))
        writer.writeheader()
        for source in sorted(sources):
            item = sources[source]
            writer.writerow({"src": source, "scene": item["scene"],
                             "label": "REAL" if item["truth"] == "0_real" else "FAKE",
                             "split": assignment[source]})
    counts = {split: sum(value == split for value in assignment.values())
              for split in ("calibration", "development", "reserved")}
    audit = {
        "scope": "Chimera real screen recapture source-level simulation split",
        "source_manifest_sha256": EXPECTED_SHA,
        "split_manifest": str(OUTPUT),
        "split_manifest_sha256": file_hash(OUTPUT),
        "rule": "SHA256(salt + UTF8 source id), sorted separately within each of six scene/label strata",
        "per_stratum": {"calibration": 40, "development": 20, "reserved": 140},
        "source_counts": counts,
        "reserved_policy": "do not inspect or tune simulator parameters on reserved sources",
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
