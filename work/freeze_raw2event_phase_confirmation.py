"""Freeze ten new, unopened Raw2Event sources for a single phase check."""

import csv
import hashlib
import json
from pathlib import Path


REGISTRY = Path("E:/ai_image_origin_research/data/manifests/raw2event_cifar_source_index.csv")
REGISTRY_SHA = "0cb84d5b31a46cfbafd427726a08e7e2c10331f1111215d3aee5ad515c48ab8b"
OLDER_SPLIT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
OUT = Path("E:/ai_image_origin_research/data/manifests/raw2event_phase_confirmation_v1.csv")
AUDIT = Path("work/raw2event_phase_confirmation_freeze.json")
SEED = "raw2event-phase-confirmation-2026-09-25-v1"


def main() -> None:
    if hashlib.sha256(REGISTRY.read_bytes()).hexdigest() != REGISTRY_SHA:
        raise RuntimeError("source registry fingerprint changed")
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    used = {row["prefix"] for row in csv.DictReader(OLDER_SPLIT.open(encoding="utf-8", newline=""))}
    if len(rows) != 5914 or len(used) != 35:
        raise RuntimeError("expected original registry and frozen earlier split")
    by_class = {}
    for row in rows:
        if row["prefix"] in used:
            continue
        digest = hashlib.sha256(f"{SEED}|{row['prefix']}".encode()).hexdigest()
        row = {**row, "role": "phase_confirmation", "selection_hash": digest}
        by_class.setdefault(row["class_name"], []).append(row)
    if len(by_class) != 10:
        raise RuntimeError("all CIFAR classes required")
    chosen = [min(by_class[name], key=lambda row: row["selection_hash"]) for name in sorted(by_class)]
    if OUT.exists():
        existing = list(csv.DictReader(OUT.open(encoding="utf-8", newline="")))
        if [row["prefix"] for row in existing] != [row["prefix"] for row in chosen]:
            raise RuntimeError("existing frozen confirmation manifest differs from deterministic selection")
        if not AUDIT.exists():
            raise RuntimeError("existing manifest has no original freeze audit")
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        if audit["manifest_sha256"] != hashlib.sha256(OUT.read_bytes()).hexdigest():
            raise RuntimeError("existing freeze audit fingerprint mismatch")
        print(json.dumps({"already_frozen": True, "manifest_sha256": audit["manifest_sha256"],
                          "count": audit["count"]}, ensure_ascii=False))
        return
    unopened = all(not Path("E:/ai_image_origin_research/data/raw/raw2event_probe", row["raw_path"]).exists()
                   and not Path("E:/ai_image_origin_research/data/raw/raw2event_probe", row["rgb_path"]).exists()
                   for row in chosen)
    with OUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) + ["role", "selection_hash"])
        writer.writeheader()
        writer.writerows(chosen)
    audit = {"registry_sha256": REGISTRY_SHA, "prior_split_count": len(used),
             "selection_rule": "lowest SHA256(seed|prefix) within each class after excluding all earlier 35 roles",
             "seed": SEED, "count": len(chosen), "manifest_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
             "prefixes": [{"prefix": row["prefix"], "class_name": row["class_name"],
                           "capture_day": row["capture_day"], "selection_hash": row["selection_hash"]}
                          for row in chosen],
             "raw_and_rgb_unopened_at_freeze": unopened}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
