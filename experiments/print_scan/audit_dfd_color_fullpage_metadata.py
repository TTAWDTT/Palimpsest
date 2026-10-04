"""Extract and inspect one complete DFD color TIFF per printer folder.

The gzipped tar must be sequentially traversed; selected 190 MB sheets live on E:.
This checks encoded scan scale without assuming the website's generic 800 ppi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from PIL import Image


ARCHIVE = Path(
    r"E:\ai_image_origin_research\data\raw\dfd_halftone_color\HalftoneImages-Color.tar.gz"
)
INVENTORY = Path("work/dfd_color_inventory.json")
DEST = Path(r"E:\ai_image_origin_research\data\derived\dfd_color_fullpage_sample")
OUT = Path("work/dfd_color_fullpage_metadata.json")
EXPECTED_BYTES = 25_818_121_374


def inspect_tiff(output: Path) -> dict:
    with Image.open(output) as image:
        x_resolution = image.tag_v2.get(282)
        y_resolution = image.tag_v2.get(283)
        resolution_unit = image.tag_v2.get(296)
        if (
            x_resolution is not None
            and y_resolution is not None
            and resolution_unit == 2
        ):
            encoded_dpi = [float(x_resolution), float(y_resolution)]
        elif (
            x_resolution is not None
            and y_resolution is not None
            and resolution_unit == 3
        ):
            encoded_dpi = [float(x_resolution) * 2.54, float(y_resolution) * 2.54]
        else:
            encoded_dpi = None
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "compression": str(image.info.get("compression")),
            "encoded_dpi": encoded_dpi,
            "pillow_interpreted_dpi": [float(v) for v in image.info["dpi"]]
            if "dpi" in image.info
            else None,
            "tiff_x_resolution": str(x_resolution),
            "tiff_y_resolution": str(y_resolution),
            "tiff_resolution_unit": str(resolution_unit),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="reinspect already SHA-recorded extracted TIFFs without rereading the 25 GB tar",
    )
    args = parser.parse_args()
    if args.metadata_only:
        report = json.loads(OUT.read_text(encoding="utf-8"))
        for record in report["records"]:
            output = Path(record["path"])
            if output.stat().st_size != record["bytes"]:
                raise ValueError(f"previously recorded TIFF size changed: {output}")
            digest = hashlib.sha256()
            with output.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"previously recorded TIFF SHA changed: {output}")
            record.update(inspect_tiff(output))
        OUT.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {row["folder"]: row["encoded_dpi"] for row in report["records"]},
                indent=2,
            ),
            flush=True,
        )
        return
    if ARCHIVE.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("verified archive missing or wrong length")
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    selected = {}
    for row in inventory["files"]:
        if row["extension"] == ".tiff" and row["folder"] not in selected:
            selected[row["folder"]] = row
    if len(selected) != 9:
        raise ValueError(f"expected nine printer folders, got {len(selected)}")
    names = {row["name"]: folder for folder, row in selected.items()}
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(ARCHIVE, mode="r|gz") as tar:
        for member in tar:
            folder = names.get(member.name)
            if folder is None:
                continue
            expected = selected[folder]
            if not member.isfile() or member.size != expected["bytes"]:
                raise ValueError(f"selected whole sheet changed: {member.name}")
            output = DEST / f"{folder}.tiff"
            digest = hashlib.sha256()
            copied = 0
            with tar.extractfile(member) as source, output.open("wb") as sink:
                while chunk := source.read(8 * 1024 * 1024):
                    sink.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
            if copied != member.size:
                output.unlink(missing_ok=True)
                raise IOError(f"truncated TIFF member: {member.name}")
            record = {
                "folder": folder,
                "archive_member": member.name,
                "path": str(output),
                "bytes": copied,
                "sha256": digest.hexdigest(),
                **inspect_tiff(output),
            }
            records.append(record)
    if len(records) != len(selected):
        raise ValueError(
            f"selected full pages missing: {len(records)} of {len(selected)}"
        )
    report = {
        "archive_sha256_from_verified_download": "dd5fa20cb13363582a440eb9a628d1311959abdc3bfcd55da06b6bbf3a1dd202",
        "selection": "first TIFF in archive order per printer folder; nine of 133 TIFFs",
        "records": records,
    }
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": len(records),
                "dpi_by_folder": {row["folder"]: row["encoded_dpi"] for row in records},
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
