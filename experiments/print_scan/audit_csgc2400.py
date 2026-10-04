"""Full-entry audit of official CSGC 2400-spi paired archive."""

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ARCHIVE = Path(r"E:\ai_image_origin_research\data\raw\csgc\CSGC_scan2400spi.zip")
OUT = Path("work/csgc2400_full_audit.json")
EXPECTED_BYTES = 113_349_148
PATTERN = re.compile(
    r"^2400dpi_NEW/(bin_exact_crop_4|resize_exact_crop)/Scan(\d{5})\.tif$"
)


def main() -> None:
    if ARCHIVE.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("CSGC archive incomplete or unexpected byte size")
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    folders = {"bin_exact_crop_4": {}, "resize_exact_crop": {}}
    with zipfile.ZipFile(ARCHIVE) as archive:
        if len(archive.infolist()) != 1903:
            raise ValueError("unexpected ZIP member count")
        for info in archive.infolist():
            if info.is_dir():
                continue
            match = PATTERN.fullmatch(info.filename)
            if not match:
                raise ValueError(f"unexpected member: {info.filename}")
            folder, item_id = match.groups()
            if item_id in folders[folder]:
                raise ValueError(f"duplicate item: {info.filename}")
            raw = archive.read(info)  # ZipFile checks each member CRC.
            with Image.open(io.BytesIO(raw)) as image:
                array = np.asarray(image)
                dpi = image.info.get("dpi")
                dpi = [float(x) for x in dpi] if dpi else None
            if array.shape != (400, 400) or array.dtype != np.uint8:
                raise ValueError(f"unexpected image format: {info.filename}")
            record = {
                "name": info.filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "encoded_dpi": dpi,
                "mean": float(array.mean() / 255),
                "std": float(array.std() / 255),
            }
            if folder == "bin_exact_crop_4":
                blocks = array.reshape(100, 4, 100, 4)
                if not np.all((array == 0) | (array == 255)) or not np.all(
                    blocks.max(axis=(1, 3)) == blocks.min(axis=(1, 3))
                ):
                    raise ValueError(
                        f"binary template is not exact 4x: {info.filename}"
                    )
                record["template_100x100_sha256"] = hashlib.sha256(
                    array[::4, ::4].tobytes()
                ).hexdigest()
            folders[folder][item_id] = record
    ids_a = set(folders["bin_exact_crop_4"])
    ids_b = set(folders["resize_exact_crop"])
    if ids_a != ids_b or len(ids_a) != 950:
        raise ValueError("binary/scan pairs do not cover the same 950 IDs")
    template_hashes = [
        r["template_100x100_sha256"] for r in folders["bin_exact_crop_4"].values()
    ]
    report = {
        "url": "https://www.univ-st-etienne.fr/graphical-code-estimation/CSGC_scan2400spi.zip",
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": digest.hexdigest(),
        "zip_members": 1903,
        "paired_ids": len(ids_a),
        "crc_checked_for_every_image": True,
        "all_binary_images_exact_4x_of_100x100_templates": True,
        "distinct_template_hashes": len(set(template_hashes)),
        "encoded_dpi_counts": {
            "binary_72dpi": sum(
                x["encoded_dpi"] == [72.0, 72.0]
                for x in folders["bin_exact_crop_4"].values()
            ),
            "scan_72dpi": sum(
                x["encoded_dpi"] == [72.0, 72.0]
                for x in folders["resize_exact_crop"].values()
            ),
        },
        "ids": sorted(ids_a),
        "records": folders,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "archive_bytes",
                    "archive_sha256",
                    "zip_members",
                    "paired_ids",
                    "distinct_template_hashes",
                    "encoded_dpi_counts",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
