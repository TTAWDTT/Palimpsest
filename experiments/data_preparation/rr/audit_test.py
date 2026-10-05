"""Validate RRDataset test images and derive source-grouped inference manifests."""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, WORK_DIR

import argparse
import csv
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageFile


ROOT = DATA_ROOT / "derived/rr_test"
ARCHIVE_MANIFEST = DATA_ROOT / "manifests/rr_test_archive_files.csv"
IMAGE_MANIFEST = DATA_ROOT / "manifests/rr_test_files.csv"
BFREE_MANIFEST = DATA_ROOT / "manifests/rr_test_bfree_manifest.csv"
UNMATCHED_MANIFEST = DATA_ROOT / "manifests/rr_test_unmatched_sources.csv"
TRAINVAL_MANIFEST = DATA_ROOT / "manifests/rr_trainval_files.csv"
SUMMARY = WORK_DIR / "rr_test_audit.json"
CONDITIONS = ("original", "transfer", "redigital")
EXPECTED_IMAGES_PER_CLASS_AND_CONDITION = 8500
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def source_id(condition: str, stem: str) -> str:
    prefix = f"{condition}_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-one-missing-redigital-real", action="store_true")
    parser.add_argument("--accept-14-exact-trainval-overlaps", action="store_true")
    args = parser.parse_args()
    SUMMARY.unlink(missing_ok=True)
    if not ROOT.is_dir() or not ARCHIVE_MANIFEST.is_file():
        raise ValueError("Verified extraction and archive manifest are required")
    with ARCHIVE_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        archive_rows = list(csv.DictReader(handle))
    if not archive_rows:
        raise ValueError("Empty archive manifest")

    with TRAINVAL_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        trainval_rows = list(csv.DictReader(handle))
    trainval_hashes = {row["sha256"] for row in trainval_rows}
    trainval_ids = {(row["label"], Path(row["filename"]).stem) for row in trainval_rows}

    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    counts = Counter()
    formats = Counter()
    suffixes = Counter()
    source_groups = defaultdict(list)
    sizes = []
    errors = []
    nonimages = []
    exact_trainval_overlap = []
    exact_trainval_overlap_sources = set()
    id_trainval_overlap = set()
    image_rows = []
    seen_paths = set()

    for index, row in enumerate(archive_rows, start=1):
        relative_path = row["relative_path"]
        if relative_path in seen_paths:
            raise ValueError(f"Duplicate path in archive manifest: {relative_path}")
        seen_paths.add(relative_path)
        path = ROOT / relative_path
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(
                f"Extracted file missing or size mismatch: {relative_path}"
            )
        parts = Path(relative_path).parts
        if (
            len(parts) != 3
            or parts[0] not in CONDITIONS
            or parts[1] not in ("ai", "real")
        ):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                errors.append(
                    {"path": relative_path, "error": "Unclassified image path"}
                )
            nonimages.append(relative_path)
            continue
        condition, label, filename = parts
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            errors.append(
                {"path": relative_path, "error": "Unknown file type in image group"}
            )
            nonimages.append(relative_path)
            continue
        digest = hashlib.sha256()
        with path.open("rb") as image_file:
            while chunk := image_file.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError(
                f"Extracted image hash differs from archive: {relative_path}"
            )
        identifier = source_id(condition, Path(filename).stem)
        key = (label, identifier)
        try:
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
                image.load()
        except Exception as exc:
            errors.append(
                {"path": relative_path, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        if not image_format:
            errors.append({"path": relative_path, "error": "No decoded image format"})
            continue
        image_row = {
            "filename": relative_path,
            "condition": condition,
            "label": label,
            "source_id": identifier,
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "width": width,
            "height": height,
            "format": image_format,
        }
        image_rows.append(image_row)
        counts[(condition, label)] += 1
        formats[(condition, label, image_format)] += 1
        suffixes[(condition, label, suffix)] += 1
        sizes.append(width * height)
        source_groups[(condition, *key)].append(relative_path)
        if row["sha256"] in trainval_hashes:
            exact_trainval_overlap.append(relative_path)
            exact_trainval_overlap_sources.add(key)
        if key in trainval_ids:
            id_trainval_overlap.add(key)
        if index % 2000 == 0:
            print(
                f"audited={index}/{len(archive_rows)} decoded={len(image_rows)}",
                flush=True,
            )

    group_keys = {
        condition: {
            (label, identifier)
            for grouped_condition, label, identifier in source_groups
            if grouped_condition == condition
        }
        for condition in CONDITIONS
    }
    duplicate_group_examples = [
        {
            "condition": condition,
            "label": label,
            "source_id": identifier,
            "count": len(paths),
        }
        for (condition, label, identifier), paths in source_groups.items()
        if len(paths) > 1
    ]
    expected_counts_met = all(
        counts[(condition, label)] == EXPECTED_IMAGES_PER_CLASS_AND_CONDITION
        for condition in CONDITIONS
        for label in ("ai", "real")
    )
    known_one_missing_pattern = all(
        counts[(condition, label)]
        == (
            EXPECTED_IMAGES_PER_CLASS_AND_CONDITION
            - int((condition, label) == ("redigital", "real"))
        )
        for condition in CONDITIONS
        for label in ("ai", "real")
    )
    all_three_groups = set.intersection(*group_keys.values())
    unmatched_source_examples = {
        condition: sorted(group_keys[condition] - all_three_groups)[:20]
        for condition in CONDITIONS
    }
    result = {
        "archive_files": len(archive_rows),
        "decoded_images": len(image_rows),
        "nominal_images_per_class_and_condition": EXPECTED_IMAGES_PER_CLASS_AND_CONDITION,
        "counts_by_condition_and_label": {
            f"{condition}/{label}": counts[(condition, label)]
            for condition in CONDITIONS
            for label in ("ai", "real")
        },
        "formats_by_condition_label_format": {
            "/".join(key): count for key, count in sorted(formats.items())
        },
        "suffixes_by_condition_label_suffix": {
            "/".join(key): count for key, count in sorted(suffixes.items())
        },
        "pixel_area": {
            "minimum": min(sizes) if sizes else None,
            "median": statistics.median(sizes) if sizes else None,
            "maximum": max(sizes) if sizes else None,
        },
        "source_groups_by_condition": {
            condition: len(group_keys[condition]) for condition in CONDITIONS
        },
        "paired_source_groups": {
            "original_transfer": len(group_keys["original"] & group_keys["transfer"]),
            "original_redigital": len(group_keys["original"] & group_keys["redigital"]),
            "all_three": len(set.intersection(*group_keys.values())),
        },
        "expected_counts_met": expected_counts_met,
        "known_one_missing_pattern": known_one_missing_pattern,
        "accepted_one_missing_exception": (
            args.accept_one_missing_redigital_real and known_one_missing_pattern
        ),
        "unmatched_source_counts": {
            condition: len(group_keys[condition] - all_three_groups)
            for condition in CONDITIONS
        },
        "unmatched_source_examples": unmatched_source_examples,
        "duplicate_condition_source_groups": len(duplicate_group_examples),
        "duplicate_condition_source_examples": duplicate_group_examples[:20],
        "exact_trainval_overlap_count": len(exact_trainval_overlap),
        "exact_trainval_overlap_examples": exact_trainval_overlap[:20],
        "exact_trainval_overlap_source_groups": len(exact_trainval_overlap_sources),
        "accepted_14_exact_trainval_overlaps": (
            args.accept_14_exact_trainval_overlaps
            and len(exact_trainval_overlap) == 14
            and len(exact_trainval_overlap_sources) == 14
        ),
        "source_id_trainval_overlap_count": len(id_trainval_overlap),
        "source_id_trainval_overlap_examples": sorted(id_trainval_overlap)[:20],
        "nonimage_files": nonimages,
        "decode_errors": errors[:50],
        "decode_error_count": len(errors),
    }
    result["audit_passed"] = False
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if (
        errors
        or duplicate_group_examples
        or id_trainval_overlap
        or (
            exact_trainval_overlap and not result["accepted_14_exact_trainval_overlaps"]
        )
        or (
            not expected_counts_met
            and not (
                args.accept_one_missing_redigital_real and known_one_missing_pattern
            )
        )
    ):
        raise ValueError("Image audit failed decoding, grouping, or split integrity")

    with UNMATCHED_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "label", "source_id"])
        writer.writeheader()
        writer.writerows(
            {"condition": condition, "label": label, "source_id": identifier}
            for condition in CONDITIONS
            for label, identifier in sorted(group_keys[condition] - all_three_groups)
        )
    ordered_rows = sorted(
        image_rows,
        key=lambda row: (
            CONDITIONS.index(row["condition"]),
            row["label"],
            row["source_id"],
        ),
    )
    temporary_image_manifest = IMAGE_MANIFEST.with_suffix(".csv.partial")
    with temporary_image_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered_rows[0]))
        writer.writeheader()
        writer.writerows(ordered_rows)
    os.replace(temporary_image_manifest, IMAGE_MANIFEST)

    temporary_bfree_manifest = BFREE_MANIFEST.with_suffix(".csv.partial")
    with temporary_bfree_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "src", "label", "B-Free"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "filename": row["filename"],
                "src": f"{row['label']}/{row['source_id']}",
                "label": "FAKE" if row["label"] == "ai" else "REAL",
                "B-Free": "",
            }
            for row in ordered_rows
        )
    os.replace(temporary_bfree_manifest, BFREE_MANIFEST)
    result["audit_passed"] = True
    result["bfree_manifest_sha256"] = hashlib.sha256(
        BFREE_MANIFEST.read_bytes()
    ).hexdigest()
    result["archive_manifest_sha256"] = hashlib.sha256(
        ARCHIVE_MANIFEST.read_bytes()
    ).hexdigest()
    temporary_summary = SUMMARY.with_suffix(".json.partial")
    temporary_summary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_summary, SUMMARY)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
