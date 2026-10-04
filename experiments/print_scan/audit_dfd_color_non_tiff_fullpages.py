"""Verify DFD color's six exceptional whole-page PNG scans and encoded DPI."""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, WORK_DIR

import hashlib
import json
import tarfile
from pathlib import Path

from PIL import Image


ARCHIVE = DATA_ROOT / "raw/dfd_halftone_color/HalftoneImages-Color.tar.gz"
INVENTORY = WORK_DIR / "dfd_color_inventory.json"
OUT = WORK_DIR / "dfd_color_non_tiff_fullpages.json"
DEST = DATA_ROOT / "derived/dfd_color_fullpage_sample/png_exceptions"
EXPECTED_BYTES = 25_818_121_374


def main() -> None:
    if ARCHIVE.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("archive not verified at official byte count")
    files = json.loads(INVENTORY.read_text(encoding="utf-8"))["files"]
    selected = {
        row["name"]: row
        for row in files
        if row["extension"] == ".png"
        and Path(row["name"]).name.startswith("full_600_600_600_110216_")
    }
    if len(selected) != 6:
        raise ValueError(
            f"expected six exceptional PNG full pages; got {len(selected)}"
        )
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(ARCHIVE, mode="r|gz") as tar:
        for member in tar:
            if member.name not in selected:
                continue
            expected = selected[member.name]
            if not member.isfile() or member.size != expected["bytes"]:
                raise ValueError(f"member changed: {member.name}")
            out = DEST / Path(member.name).name
            digest = hashlib.sha256()
            count = 0
            with tar.extractfile(member) as source, out.open("wb") as sink:
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
                    sink.write(block)
                    count += len(block)
            if count != member.size:
                out.unlink(missing_ok=True)
                raise IOError(f"truncated member: {member.name}")
            with Image.open(out) as image:
                dpi = image.info.get("dpi")
                records.append(
                    {
                        "archive_member": member.name,
                        "path": str(out),
                        "bytes": count,
                        "sha256": digest.hexdigest(),
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                        "encoded_dpi": [float(v) for v in dpi] if dpi else None,
                    }
                )
            if len(records) == len(selected):
                break
    if len(records) != 6:
        raise ValueError(f"archive yielded only {len(records)} selected PNGs")
    report = {
        "selection": "all six PNGs with full_600_600_600_110216 name prefix",
        "records": records,
    }
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "count": len(records),
                "encoded_dpi": [row["encoded_dpi"] for row in records],
                "dimensions": [(row["width"], row["height"]) for row in records],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
