"""Inventory the verified DFD color archive without extracting 25 GB of scans.

Run only after download_dfd_color.ps1 has promoted the exact-size partial file
to a verified final name. Archive names are data, never filesystem targets.
"""

from __future__ import annotations

import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ARCHIVE = Path(r"E:\ai_image_origin_research\data\raw\dfd_halftone_color\HalftoneImages-Color.tar.gz")
OUT = Path("work/dfd_color_inventory.json")
EXPECTED_BYTES = 25_818_121_374


def main() -> None:
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("DFD color archive has not been promoted at official byte count")
    rows = []
    folder_counts = Counter()
    format_counts = Counter()
    file_sizes = defaultdict(list)
    dirs = 0
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
            rows.append({"name": member.name, "bytes": member.size,
                         "folder": folder, "extension": extension})
            folder_counts[folder] += 1
            format_counts[extension] += 1
            file_sizes[folder].append(member.size)
    result = {
        "archive": str(ARCHIVE), "archive_bytes": EXPECTED_BYTES,
        "integrity_prior": "download_dfd_color.ps1 .NET SHA-256 and full tar/gzip traversal; inspect its log",
        "file_count": len(rows), "directory_count": dirs,
        "file_format_counts": dict(format_counts),
        "folder_counts": dict(sorted(folder_counts.items())),
        "folder_member_bytes_min_max": {key: [min(value), max(value)]
                                        for key, value in sorted(file_sizes.items())},
        "files": rows,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("file_count", "directory_count", "file_format_counts", "folder_counts")},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
