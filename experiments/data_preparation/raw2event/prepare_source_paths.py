"""Audit whether Raw2Event filenames deterministically identify CIFAR-10 sources.

This reads a pinned Hugging Face directory listing and the MD5-verified official
CIFAR-10 archive. It does not download any additional capture videos.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import tarfile
import warnings

import numpy as np
import requests

from experiments.data_preparation.raw2event.prepare_download_cifar10_python import (
    TARGET as CIFAR_ARCHIVE,
    md5 as file_md5,
)


REPO = "raw2event/raw2event"
REVISION = "9df99d9ed09e5ed705f50cae011e49cb2af620b9"
ROOT_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}"
REGISTRY = DATA_ROOT / "manifests/raw2event_cifar_source_index.csv"
SUMMARY = WORK_DIR / "raw2event_cifar_source_index_audit.json"
NAME = re.compile(r"^(\d+)_([a-z]+)_([^_]+)_(\d+)_(\d{8})_(\d{6})\.mkv$")
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def list_files(directory: str) -> list[dict]:
    session = requests.Session()
    allowed_root = f"{ROOT_URL}/{directory}"
    url = allowed_root
    params = {"limit": 1000}
    items = []
    page = 0
    while url:
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise RuntimeError("unexpected HF directory listing")
        items.extend(batch)
        page += 1
        if page % 10 == 0:
            print(f"{directory}: listed {page} pages / {len(items)} paths", flush=True)
        url = response.links.get("next", {}).get("url")
        if url and not url.startswith(allowed_root):
            raise RuntimeError("pagination escaped pinned directory URL")
        params = None
    return items


def official_labels() -> dict[str, np.ndarray]:
    if (
        CIFAR_ARCHIVE.stat().st_size != 170_498_071
        or file_md5(CIFAR_ARCHIVE) != "c58f30108f718f92721af3b95e74349a"
    ):
        raise RuntimeError("CIFAR archive does not match University of Toronto MD5")
    result = {}
    with tarfile.open(CIFAR_ARCHIVE, "r:gz") as archive:
        for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
            with archive.extractfile(f"cifar-10-batches-py/{name}") as stream:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=r"dtype\(\): align should be passed.*"
                    )
                    payload = pickle.load(io.BytesIO(stream.read()), encoding="bytes")
            labels = np.asarray(payload[b"labels"], dtype=np.int8)
            if len(labels) != 10000:
                raise RuntimeError(f"unexpected label count for {name}")
            result[name] = labels
    return result


def parse_paths(items: list[dict]) -> tuple[list[dict], list[str]]:
    rows, bad = [], []
    seen = set()
    for item in items:
        path = item.get("path", "")
        if item.get("type") != "file" or not path.startswith("frames_raw/"):
            bad.append(path)
            continue
        stem = Path(path).name
        match = NAME.fullmatch(stem)
        if not match:
            bad.append(path)
            continue
        numeric_id, class_name, subid, row_index, day, clock = match.groups()
        if stem in seen:
            raise RuntimeError(f"duplicate basename: {stem}")
        seen.add(stem)
        rows.append(
            {
                "prefix": stem[:-4],
                "raw_path": path,
                "id": numeric_id,
                "class_name": class_name,
                "subid": subid,
                "row": int(row_index),
                "capture_day": day,
                "capture_clock": clock,
            }
        )
    return rows, bad


def main() -> None:
    items = list_files("frames_raw")
    rgb_items = list_files("frames_rgb")
    meta_items = list_files("meta_raw")
    rows, bad_paths = parse_paths(items)
    raw_prefixes = {row["prefix"] for row in rows}
    rgb_prefixes = {
        Path(item["path"]).stem for item in rgb_items if item.get("type") == "file"
    }
    meta_prefixes = {
        Path(item["path"]).stem for item in meta_items if item.get("type") == "file"
    }
    labels = official_labels()
    by_subid = defaultdict(list)
    for row in rows:
        by_subid[row["subid"]].append(row)
    candidates = {}
    mapping = {}
    for subid, group in sorted(by_subid.items()):
        scores = {}
        for batch, lab in labels.items():
            valid = [
                row
                for row in group
                if row["row"] < 10000 and row["class_name"] in CLASSES
            ]
            matches = sum(
                int(lab[row["row"]]) == CLASSES.index(row["class_name"])
                for row in valid
            )
            scores[batch] = {
                "valid_rows": len(valid),
                "label_matches": matches,
                "fraction": matches / len(valid) if valid else None,
            }
        candidates[subid] = scores
        ranked = sorted(
            scores, key=lambda batch: scores[batch]["label_matches"], reverse=True
        )
        if scores[ranked[0]]["fraction"] == 1.0 and (
            len(ranked) == 1
            or scores[ranked[1]]["label_matches"] < scores[ranked[0]]["label_matches"]
        ):
            mapping[subid] = ranked[0]
    for row in rows:
        batch = mapping.get(row["subid"])
        row["cifar_batch"] = batch or ""
        row["cifar_source_key"] = f"{batch}:{row['row']}" if batch else ""
        row["rgb_path"] = (
            f"frames_rgb/{row['prefix']}.mkv" if row["prefix"] in rgb_prefixes else ""
        )
        row["meta_path"] = (
            f"meta_raw/{row['prefix']}.dat" if row["prefix"] in meta_prefixes else ""
        )
        row["class_label_verified"] = (
            batch is not None
            and row["row"] < 10000
            and row["class_name"] in CLASSES
            and int(labels[batch][row["row"]]) == CLASSES.index(row["class_name"])
        )
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prefix",
        "raw_path",
        "rgb_path",
        "meta_path",
        "id",
        "class_name",
        "subid",
        "row",
        "cifar_batch",
        "cifar_source_key",
        "class_label_verified",
        "capture_day",
        "capture_clock",
    ]
    with REGISTRY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    summary = {
        "repo": REPO,
        "revision": REVISION,
        "scope": "three direct per-prefix directories, not tar shards",
        "file_entries": len(items),
        "parsed_prefixes": len(rows),
        "bad_paths_count": len(bad_paths),
        "bad_paths_first20": bad_paths[:20],
        "rgb_entries": len(rgb_items),
        "meta_entries": len(meta_items),
        "raw_with_rgb_and_meta": len(raw_prefixes & rgb_prefixes & meta_prefixes),
        "raw_missing_rgb": sorted(raw_prefixes - rgb_prefixes),
        "raw_missing_meta": sorted(raw_prefixes - meta_prefixes),
        "rgb_not_raw_count": len(rgb_prefixes - raw_prefixes),
        "meta_not_raw_count": len(meta_prefixes - raw_prefixes),
        "subid_counts": dict(Counter(row["subid"] for row in rows)),
        "class_counts": dict(Counter(row["class_name"] for row in rows)),
        "capture_day_counts": dict(Counter(row["capture_day"] for row in rows)),
        "candidate_batch_label_consistency": candidates,
        "inferred_batch_mapping": mapping,
        "label_verified_prefixes": sum(row["class_label_verified"] for row in rows),
        "unresolved_prefixes": sum(not row["class_label_verified"] for row in rows),
        "unique_cifar_source_keys": len(
            {
                (row["cifar_batch"], row["row"])
                for row in rows
                if row["class_label_verified"]
            }
        ),
        "registry_path": str(REGISTRY),
        "registry_sha256": digest,
        "qualification": "filename and class-label consistency; only two pixel contents directly searched against CIFAR",
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "file_entries",
                    "parsed_prefixes",
                    "bad_paths_count",
                    "rgb_entries",
                    "meta_entries",
                    "raw_with_rgb_and_meta",
                    "subid_counts",
                    "inferred_batch_mapping",
                    "label_verified_prefixes",
                    "unresolved_prefixes",
                    "unique_cifar_source_keys",
                    "registry_sha256",
                )
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
