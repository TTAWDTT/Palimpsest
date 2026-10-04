"""Safely extract the verified RRDataset test archive and index every file."""

from __future__ import annotations

import csv
import hashlib
import os
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


ARCHIVE = Path("E:/ai_image_origin_research/data/raw/RRDataset_test.tar.gz")
DESTINATION = Path("E:/ai_image_origin_research/data/derived/rr_test")
STAGING = DESTINATION.with_name("rr_test.incomplete")
MANIFEST = Path("E:/ai_image_origin_research/data/manifests/rr_test_archive_files.csv")
ROOT_NAME = "RRDataset_final"
EXPECTED_MD5 = "13c3ff3d61986170cc0c8cf76a35cd4b"
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def verify_archive() -> None:
    if ARCHIVE.stat().st_size != 20_117_869_400:
        raise ValueError("Archive size differs from the author's test package")
    digest = hashlib.md5()
    with ARCHIVE.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_MD5:
        raise ValueError("Archive MD5 differs from the author's test package")


def safe_target(member_name: str) -> Path:
    source = PurePosixPath(member_name)
    if source.is_absolute() or ".." in source.parts or not source.parts:
        raise ValueError(f"Unsafe archive path: {member_name}")
    if source.parts[0] != ROOT_NAME:
        raise ValueError(f"Unexpected archive root: {member_name}")
    for part in source.parts[1:]:
        if (
            any(character in part for character in '<>:"\\|?*')
            or part.rstrip(" .") != part
            or part.split(".", 1)[0].upper() in RESERVED_NAMES
        ):
            raise ValueError(f"Unsafe Windows archive name: {member_name}")
    target = STAGING.joinpath(*source.parts[1:])
    if not target.resolve().is_relative_to(STAGING.resolve()):
        raise ValueError(f"Archive member leaves destination: {member_name}")
    return target


def main() -> None:
    verify_archive()
    if STAGING.exists() or DESTINATION.exists():
        raise ValueError("Extraction target already exists; inspect it before retrying")
    STAGING.mkdir(parents=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    partial_manifest = MANIFEST.with_suffix(".csv.partial")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    total_bytes = 0
    with partial_manifest.open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.DictWriter(
            index_file,
            fieldnames=["archive_name", "relative_path", "bytes", "sha256"],
        )
        writer.writeheader()
        with tarfile.open(ARCHIVE, mode="r|gz") as archive:
            for member in archive:
                target = safe_target(member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"Unsupported archive entry: {member.name}")
                relative_path = target.relative_to(STAGING).as_posix()
                normalized_path = relative_path.casefold()
                if normalized_path in seen:
                    raise ValueError(f"Duplicate archive member: {relative_path}")
                seen.add(normalized_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                content = archive.extractfile(member)
                if content is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                digest = hashlib.sha256()
                actual_bytes = 0
                with content, target.open("wb") as output:
                    while chunk := content.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        actual_bytes += len(chunk)
                if actual_bytes != member.size:
                    raise ValueError(f"Size mismatch: {member.name}")
                writer.writerow(
                    {
                        "archive_name": member.name,
                        "relative_path": relative_path,
                        "bytes": actual_bytes,
                        "sha256": digest.hexdigest(),
                    }
                )
                total_bytes += actual_bytes
                condition = relative_path.split("/", 1)[0]
                counts[condition] += 1
                if len(seen) % 1000 == 0:
                    index_file.flush()
                    print(
                        f"files={len(seen)} bytes={total_bytes} conditions={dict(counts)}",
                        flush=True,
                    )
    STAGING.rename(DESTINATION)
    os.replace(partial_manifest, MANIFEST)
    print(
        f"COMPLETE files={len(seen)} bytes={total_bytes} conditions={dict(counts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
