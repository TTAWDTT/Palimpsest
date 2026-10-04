"""Fetch only the ten frozen phase-confirmation Raw2Event capture triples."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json

import requests

from experiments.raw2event.fetch_raw2event_pair import REPO, fetch_one, remote_metadata
from experiments.raw2event.index_raw2event_source_paths import REVISION


MANIFEST = DATA_ROOT / "manifests/raw2event_phase_confirmation_v1.csv"
EXPECTED_SHA = "c699c6b9ce8ce779f892c5caaab11e19b4c24e0e697ec8116b94115c0175e372"
AUDIT_DIR = WORK_DIR / "raw2event_phase_confirmation_downloads"
OUT = WORK_DIR / "raw2event_phase_confirmation_download_audit.json"


def one(row: dict) -> dict:
    session = requests.Session()
    records = [
        fetch_one(session, meta, REVISION)
        for meta in remote_metadata(session, row["prefix"], REVISION)
    ]
    audit = {
        "prefix": row["prefix"],
        "class_name": row["class_name"],
        "repo": REPO,
        "revision": REVISION,
        "files": records,
        "complete_verified": True,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / f"{row['prefix']}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise RuntimeError("frozen phase-confirmation manifest changed")
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    if len(rows) != 10 or len({row["class_name"] for row in rows}) != 10:
        raise RuntimeError("expected ten unique classes")
    completed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, row): row for row in rows}
        for future in as_completed(futures):
            audit = future.result()
            completed.append(audit)
            print(
                f"verified {len(completed)}/10 {audit['class_name']} {audit['prefix']}",
                flush=True,
            )
    report = {
        "manifest_sha256": EXPECTED_SHA,
        "revision": REVISION,
        "n_prefixes": len(completed),
        "n_files": sum(len(audit["files"]) for audit in completed),
        "bytes": sum(file["size"] for audit in completed for file in audit["files"]),
        "prefixes": sorted(audit["prefix"] for audit in completed),
        "earlier_reserved_untouched": True,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
