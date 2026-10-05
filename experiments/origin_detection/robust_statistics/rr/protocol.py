"""Registered RR source roles, inputs and feature-search limits."""

import csv
import json
import tomllib

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import DATA_ROOT, REPO_ROOT, WORK_DIR

CONFIG = REPO_ROOT / "configs/evaluation/rr_robust_statistics.toml"
ROOT = DATA_ROOT / "derived/rr_test"
REGISTRY = DATA_ROOT / "manifests/rr_simulation_source_registry.csv"
MANIFEST = DATA_ROOT / "manifests/rr_test_files.csv"
TRAINVAL = DATA_ROOT / "manifests/rr_trainval_files.csv"
RESULT = WORK_DIR / "robust_statistics/rr_first_iteration"
PINS = {
    MANIFEST: "1b5ca7632034680bbffcb8f74524ee4225ab5bb11aeead79f7fce244928501e5",
    REGISTRY: "1361f527e7b142530ff1363b24271cf40932cf120df701b3ec6cef6531b7fee5",
}


def settings():
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verified_inventory():
    audit = json.loads((RESULT / "source_audit.json").read_text(encoding="utf-8"))
    for filename, fingerprint in audit["inputs"].items():
        if file_sha256(REPO_ROOT / filename) != fingerprint:
            raise ValueError(f"Source audit input changed: {filename}")
    path = RESULT / "development_images.csv"
    if file_sha256(path) != audit["development_manifest_sha256"]:
        raise ValueError("Development inventory changed")
    rows = read_csv(path)
    if len(rows) != audit["development_images"]:
        raise ValueError("Incomplete development inventory")
    return rows, audit
