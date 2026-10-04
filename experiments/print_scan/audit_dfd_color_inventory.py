"""Inventory the verified DFD color archive without extracting 25 GB of scans.

Run only after download_dfd_color.ps1 has promoted the exact-size partial file
to a verified final name. Archive names are data, never filesystem targets.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, WORK_DIR

import json
import hashlib
import tarfile
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image


ARCHIVE = DATA_ROOT / "raw/dfd_halftone_color/HalftoneImages-Color.tar.gz"
OUT = WORK_DIR / "dfd_color_inventory.json"
EXPECTED_BYTES = 25_818_121_374
SAMPLE_DIR = DATA_ROOT / "derived/dfd_color_one_per_folder"
MAX_SAMPLE_BYTES = 384 * 1024 * 1024


def copy_sample(source, destination: Path, expected_bytes: int) -> str:
    digest = hashlib.sha256()
    copied = 0
    with destination.open("wb") as output:
        while chunk := source.read(8 * 1024 * 1024):
            copied += len(chunk)
            output.write(chunk)
            digest.update(chunk)
    if copied != expected_bytes:
        destination.unlink(missing_ok=True)
        raise IOError(f"selected member was truncated: {destination}")
    return digest.hexdigest()


def main() -> None:
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            "DFD color archive has not been promoted at official byte count"
        )
    rows = []
    folder_counts = Counter()
    format_counts = Counter()
    file_sizes = defaultdict(list)
    samples = []
    sampled_folders = set()
    dirs = 0
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, mode="r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive path: {member.name}")
            if member.isdir():
                dirs += 1
                continue
            if not member.isfile():
                raise ValueError(f"unexpected archive member type: {member.name}")
            extension = path.suffix.lower()
            folder = path.parts[1] if len(path.parts) > 2 else "<root>"
            rows.append(
                {
                    "name": member.name,
                    "bytes": member.size,
                    "folder": folder,
                    "extension": extension,
                }
            )
            folder_counts[folder] += 1
            format_counts[extension] += 1
            file_sizes[folder].append(member.size)
            if (
                folder != "<root>"
                and folder not in sampled_folders
                and extension in {".tif", ".tiff", ".png"}
                and member.size <= MAX_SAMPLE_BYTES
            ):
                safe_folder = re.sub(r"[^A-Za-z0-9_.-]", "_", folder)
                destination = SAMPLE_DIR / f"{safe_folder}{extension}"
                with archive.extractfile(member) as source:
                    sha = copy_sample(source, destination, member.size)
                with Image.open(destination) as image:
                    dpi = image.info.get("dpi")
                    sample = {
                        "member_name": member.name,
                        "path": str(destination),
                        "bytes": member.size,
                        "sha256": sha,
                        "width": image.width,
                        "height": image.height,
                        "format": image.format,
                        "mode": image.mode,
                        "encoded_dpi": [float(v) for v in dpi]
                        if dpi is not None
                        else None,
                    }
                samples.append(sample)
                sampled_folders.add(folder)
    result = {
        "archive": str(ARCHIVE),
        "archive_bytes": EXPECTED_BYTES,
        "integrity_prior": "download_dfd_color.ps1 .NET SHA-256 and full tar/gzip traversal; inspect its log",
        "file_count": len(rows),
        "directory_count": dirs,
        "file_format_counts": dict(format_counts),
        "folder_counts": dict(sorted(folder_counts.items())),
        "folder_member_bytes_min_max": {
            key: [min(value), max(value)] for key, value in sorted(file_sizes.items())
        },
        "files": rows,
        "selected_samples": samples,
    }
    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "file_count",
                    "directory_count",
                    "file_format_counts",
                    "folder_counts",
                )
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
