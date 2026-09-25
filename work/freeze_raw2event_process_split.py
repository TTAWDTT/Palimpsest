"""Freeze a small, content-disjoint process-calibration split before more videos.

The two previously inspected prefixes are explicitly exploratory and excluded
from all new calibration/development/reserved roles.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


SOURCE = Path("E:/ai_image_origin_research/data/manifests/raw2event_cifar_source_index.csv")
SOURCE_SHA256 = "0cb84d5b31a46cfbafd427726a08e7e2c10331f1111215d3aee5ad515c48ab8b"
OUTPUT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
AUDIT = Path("work/raw2event_process_split_v1_audit.json")
SALT = "raw2event-screen-process-v1-2026-09-25"
EXPLORATORY = {
    "10000_automobile_5_1087_20251224_105416",
    "1000_airplane_1_9934_20251222_161953",
}


def rank(prefix: str) -> str:
    return hashlib.sha256(f"{SALT}|{prefix}".encode("utf-8")).hexdigest()


def main() -> None:
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("source registry fingerprint changed")
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8", newline="")))
    if len(rows) != 5914 or not all(row["class_label_verified"] == "True" for row in rows):
        raise RuntimeError("source registry is not the audited 5914-source version")
    by_class = defaultdict(list)
    for row in rows:
        if row["prefix"] not in EXPLORATORY:
            by_class[row["class_name"]].append(row)
    chosen = []
    primary_days = {}
    for class_name, group in sorted(by_class.items()):
        counts = Counter(row["capture_day"] for row in group)
        primary_day = sorted(counts, key=lambda day: (-counts[day], day))[0]
        primary_days[class_name] = primary_day
        primary = sorted((row for row in group if row["capture_day"] == primary_day),
                         key=lambda row: rank(row["prefix"]))
        if len(primary) < 3:
            raise RuntimeError(f"insufficient primary-day sources in {class_name}")
        for role, row in zip(("calibration", "development", "reserved"), primary[:3]):
            chosen.append({**row, "role": role, "selection_hash": rank(row["prefix"])})
        secondary = sorted((row for row in group if row["capture_day"] != primary_day),
                           key=lambda row: rank(row["prefix"]))
        if secondary:
            chosen.append({**secondary[0], "role": "cross_day_stress_reserved",
                           "selection_hash": rank(secondary[0]["prefix"])})
    for prefix in sorted(EXPLORATORY):
        row = next(row for row in rows if row["prefix"] == prefix)
        chosen.append({**row, "role": "exploratory_already_seen", "selection_hash": rank(prefix)})
    if len({row["cifar_source_key"] for row in chosen}) != len(chosen):
        raise RuntimeError("source overlap among frozen roles")
    fields = list(rows[0]) + ["role", "selection_hash"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)
    role_counts = dict(Counter(row["role"] for row in chosen))
    summary = {"source_registry_sha256": SOURCE_SHA256, "split_path": str(OUTPUT),
               "split_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
               "salt": SALT, "roles": role_counts, "primary_day_by_class": primary_days,
               "source_disjoint": True,
               "qualification": "cross-class and cross-day effects are confounded in this corpus; reserved roles are unopened"}
    AUDIT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
