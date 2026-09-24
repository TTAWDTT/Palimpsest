"""Safely extract and index the verified RRDataset train/validation archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


ROOT_NAME = "RRDataset_original_train_val"
SPLITS = {"train", "val"}
CLASSES = {"ai", "real"}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def checked_image_path(member: tarfile.TarInfo) -> tuple[str, str, str]:
    name = member.name.replace("\\", "/")
    path = PurePosixPath(name)
    if name.startswith("/") or ".." in path.parts or len(path.parts) != 4:
        raise ValueError(f"Unexpected archive path: {member.name}")
    root, split, label, basename = path.parts
    if root != ROOT_NAME or split not in SPLITS or label not in CLASSES:
        raise ValueError(f"Unexpected archive group: {member.name}")
    if PurePosixPath(basename).suffix.lower() not in EXTENSIONS:
        raise ValueError(f"Unexpected image extension: {member.name}")
    return name, split, label


def stream_extract(
    source, target: Path, expected_size: int
) -> tuple[str, int]:
    digest = hashlib.sha256()
    copied = 0
    temporary = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as output:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            output.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
    if copied != expected_size:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Size mismatch for {target}: {copied} != {expected_size}")
    temporary.replace(target)
    return digest.hexdigest(), copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    extract_root = args.extract_root.resolve()
    seen = set()
    counts = Counter()
    records = []
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Nonregular archive member: {member.name}")
            name, split, label = checked_image_path(member)
            if name in seen:
                raise ValueError(f"Repeated archive path: {name}")
            seen.add(name)
            target = (extract_root / name).resolve()
            if not target.is_relative_to(extract_root):
                raise ValueError(f"Archive path leaves extraction root: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive image: {name}")
            with source:
                sha256, file_bytes = stream_extract(source, target, member.size)
            with Image.open(target) as image:
                width, height = image.size
                image_format = image.format
                image.verify()
            counts[f"{split}/{label}"] += 1
            records.append({
                "filename": name,
                "split": split,
                "label": label,
                "bytes": file_bytes,
                "sha256": sha256,
                "width": width,
                "height": height,
                "format": image_format,
            })

    if not all(counts[f"{split}/{label}"] for split in SPLITS for label in CLASSES):
        raise ValueError(f"Missing split/class combination: {dict(counts)}")
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {
        "archive": str(args.archive),
        "images": len(records),
        "counts": dict(sorted(counts.items())),
        "unique_filenames": len(seen),
        "total_image_bytes": sum(row["bytes"] for row in records),
        "manifest_csv": str(args.manifest_csv),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
