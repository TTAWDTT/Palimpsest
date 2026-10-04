"""Collect per-image header metadata from the fully verified DFD B/W archive.

No images are extracted or redistributed. This complements the first audit:
the official page describes 114 prints and 800 ppi, while the package mixes
TIFF and PNG files whose encoded resolutions can differ.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, WORK_DIR

import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ARCHIVE = DATA_ROOT / "raw/dfd_halftone/HalftoneImages-BW.tar.gz"
OUT = WORK_DIR / "dfd_halftone_image_metadata.json"
EXPECTED_BYTES = 4_259_915_309


def main() -> None:
    if ARCHIVE.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("archive byte count differs from verified DFD download")
    records = []
    with tarfile.open(ARCHIVE, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(
                (".tif", ".tiff", ".png")
            ):
                continue
            with tar.extractfile(member) as source:
                data = source.read()
            if len(data) != member.size:
                raise ValueError(f"truncated entry: {member.name}")
            with Image.open(io.BytesIO(data)) as image:
                dpi = image.info.get("dpi")
                if dpi is not None:
                    dpi = [float(v) for v in dpi]
                records.append(
                    {
                        "name": member.name,
                        "bytes": member.size,
                        "extension": Path(member.name).suffix.lower(),
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                        "format": image.format,
                        "dpi": dpi,
                    }
                )
    counts = Counter((r["extension"], str(r["dpi"])) for r in records)
    folders = defaultdict(Counter)
    for record in records:
        parts = Path(record["name"]).parts
        folders[parts[1]][record["extension"]] += 1
    report = {
        "archive_sha256_from_verified_audit": json.loads(
            (WORK_DIR / "dfd_halftone_audit.json").read_text(encoding="utf-8")
        )["archive_sha256"],
        "count": len(records),
        "extension_dpi_counts": [
            {"extension": ext, "dpi": dpi, "count": count}
            for (ext, dpi), count in sorted(counts.items())
        ],
        "folder_format_counts": {
            folder: dict(count) for folder, count in sorted(folders.items())
        },
        "records": records,
    }
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("count", "extension_dpi_counts", "folder_format_counts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
