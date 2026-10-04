"""Download only frozen calibration/development Raw2Event capture prefixes."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import requests

from experiments.data_preparation.raw2event.prepare_download_pair import (
    REPO,
    remote_metadata,
    fetch_one,
)
from experiments.data_preparation.raw2event.prepare_source_paths import REVISION


SPLIT = DATA_ROOT / "manifests/raw2event_process_split_v1.csv"
EXPECTED_SHA = "471fbff8020f88c1b73664e5794285df28da4792e5fb77bdfe682b5ee9bb7a43"
AUDIT_DIR = WORK_DIR / "raw2event_process_split_downloads"
SUMMARY = WORK_DIR / "raw2event_process_split_download_audit.json"


def one(prefix: str, role: str) -> dict:
    session = requests.Session()
    entries = remote_metadata(session, prefix, REVISION)
    records = [fetch_one(session, metadata, REVISION) for metadata in entries]
    audit = {
        "repo": REPO,
        "revision": REVISION,
        "prefix": prefix,
        "role": role,
        "files": records,
        "complete_verified": True,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / f"{prefix}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    if hashlib.sha256(SPLIT.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise RuntimeError("frozen split fingerprint changed")
    rows = list(csv.DictReader(SPLIT.open(encoding="utf-8", newline="")))
    selected = [
        (row["prefix"], row["role"])
        for row in rows
        if row["role"] in ("calibration", "development")
    ]
    if len(selected) != 20:
        raise RuntimeError("expected exactly 20 calibration/development prefixes")
    completed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(one, prefix, role): (prefix, role) for prefix, role in selected
        }
        for future in as_completed(futures):
            prefix, role = futures[future]
            audit = future.result()
            completed.append(audit)
            print(f"verified {len(completed)}/20: {role} {prefix}", flush=True)
    summary = {
        "repo": REPO,
        "revision": REVISION,
        "split_sha256": EXPECTED_SHA,
        "prefix_count": len(completed),
        "file_count": sum(len(x["files"]) for x in completed),
        "bytes": sum(record["size"] for x in completed for record in x["files"]),
        "prefixes": [
            {"prefix": x["prefix"], "role": x["role"]}
            for x in sorted(completed, key=lambda x: x["prefix"])
        ],
        "reserved_unopened": True,
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("prefix_count", "file_count", "bytes", "reserved_unopened")
            }
        )
    )


if __name__ == "__main__":
    main()
