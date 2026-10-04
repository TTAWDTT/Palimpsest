"""Audit paired DESCAN-18K validation/test archives without extracting them."""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.rr.analyze_rr_simulation_inputs import image_metrics, quantiles


BASE = REPO_ROOT
ARCHIVE_ROOT = Path(r"E:\ai_image_origin_research\data\raw\descan18k")
OUTPUT = BASE / "work" / "descan_pair_audit.json"


def read_rgb(archive: zipfile.ZipFile, member: str) -> tuple[Image.Image, str, str]:
    data = archive.read(member)
    fingerprint = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as opened:
        image = opened.convert("RGB")
    pixel_fingerprint = hashlib.sha256(image.tobytes()).hexdigest()
    return image, fingerprint, pixel_fingerprint


def normalized(image: Image.Image) -> np.ndarray:
    return (
        np.asarray(image.resize((128, 128), Image.Resampling.BICUBIC), dtype=np.float32)
        / 255
    )


def audit_split(split: str) -> dict:
    archive_path = ARCHIVE_ROOT / f"{split}.zip"
    by_scanner = defaultdict(lambda: defaultdict(list))
    clean_fingerprints = {}
    clean_pixel_fingerprints = {}
    dimensions = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        clean = {
            Path(name).name: name
            for name in archive.namelist()
            if name.startswith(f"{split}/clean/") and name.lower().endswith(".tif")
        }
        scan = {
            Path(name).name: name
            for name in archive.namelist()
            if name.startswith(f"{split}/scan/") and name.lower().endswith(".tif")
        }
        if len(clean) != 360 or len(scan) != 360 or clean.keys() != scan.keys():
            raise ValueError(
                f"invalid {split} pairing: {len(clean)} clean, {len(scan)} scan"
            )
        metadata = [
            json.loads(line)
            for line in archive.read(f"{split}/metadata.jsonl").splitlines()
        ]
        metadata_pairs = {
            (row["input_file_name"], row["output_file_name"]) for row in metadata
        }
        expected_pairs = {
            (f"scan/{basename}", f"clean/{basename}") for basename in clean
        }
        if len(metadata) != 360 or metadata_pairs != expected_pairs:
            raise ValueError(f"{split} metadata does not match scan/clean files")
        for basename in sorted(clean):
            scanner = basename.split("_")[0]
            clean_image, clean_sha, clean_pixel_sha = read_rgb(archive, clean[basename])
            scan_image, _, _ = read_rgb(archive, scan[basename])
            dimensions[(clean_image.size, scan_image.size)] += 1
            clean_fingerprints[basename] = clean_sha
            clean_pixel_fingerprints[basename] = clean_pixel_sha
            values = image_metrics(normalized(clean_image), normalized(scan_image))
            for feature, value in values.items():
                by_scanner[scanner][feature].append(value)
    return {
        "pairs": len(clean_fingerprints),
        "scanner_counts": {
            scanner: len(next(iter(features.values())))
            for scanner, features in by_scanner.items()
        },
        "dimensions": [
            {"clean": list(clean_size), "scan": list(scan_size), "count": count}
            for (clean_size, scan_size), count in dimensions.items()
        ],
        "feature_quantiles_by_scanner": {
            scanner: {
                feature: quantiles(values) for feature, values in features.items()
            }
            for scanner, features in by_scanner.items()
        },
        "clean_sha256_by_filename": clean_fingerprints,
        "clean_pixel_sha256_by_filename": clean_pixel_fingerprints,
    }


def main() -> None:
    audit = {split: audit_split(split) for split in ("Valid", "Test")}
    valid_hashes = set(audit["Valid"]["clean_sha256_by_filename"].values())
    test_hashes = set(audit["Test"]["clean_sha256_by_filename"].values())
    audit["cross_split_exact_clean_overlap"] = len(valid_hashes & test_hashes)
    valid_pixels = set(audit["Valid"]["clean_pixel_sha256_by_filename"].values())
    test_pixels = set(audit["Test"]["clean_pixel_sha256_by_filename"].values())
    audit["cross_split_exact_clean_pixel_overlap"] = len(valid_pixels & test_pixels)
    OUTPUT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                split: {
                    key: value
                    for key, value in audit[split].items()
                    if key
                    not in {
                        "clean_sha256_by_filename",
                        "clean_pixel_sha256_by_filename",
                    }
                }
                for split in ("Valid", "Test")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("cross_split_exact_clean_overlap:", audit["cross_split_exact_clean_overlap"])
    print(
        "cross_split_exact_clean_pixel_overlap:",
        audit["cross_split_exact_clean_pixel_overlap"],
    )


if __name__ == "__main__":
    main()
