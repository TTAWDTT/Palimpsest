"""Spot-check additional cross-SPI CSGC pair mappings by HTTP Range + CRC."""

from palimpsest.paths import WORK_DIR

import hashlib
import json

import numpy as np
from PIL import Image

from experiments.data_preparation.csgc.prepare_cross_resolution_pairs import extract


def main() -> None:
    result = {}
    for item in (101, 501, 901):
        templates = {}
        item_result = {}
        for spi, factor in (("2400", 4), ("4800", 8), ("9600", 16)):
            binary_path, binary_meta = extract(spi, f"bin_exact_crop_{factor}", item)
            scan_path, scan_meta = extract(spi, "resize_exact_crop", item)
            with Image.open(binary_path) as image:
                binary = np.asarray(image, dtype=np.uint8)
            with Image.open(scan_path) as image:
                scan = np.asarray(image, dtype=np.uint8)
            if binary.shape != scan.shape or binary.shape != (
                100 * factor,
                100 * factor,
            ):
                raise RuntimeError(f"{spi}/{item}: unexpected paired shapes")
            blocks = binary.reshape(100, factor, 100, factor)
            if not np.all(blocks.max(axis=(1, 3)) == blocks.min(axis=(1, 3))):
                raise RuntimeError(f"{spi}/{item}: nonuniform reference blocks")
            template = binary[::factor, ::factor]
            templates[spi] = template
            item_result[spi] = {
                "binary_sha256": binary_meta["sha256"],
                "scan_sha256": scan_meta["sha256"],
                "template_sha256_100x100": hashlib.sha256(
                    template.tobytes()
                ).hexdigest(),
                "paired_pixel_pearson": float(
                    np.corrcoef(binary.ravel(), scan.ravel())[0, 1]
                ),
            }
        item_result["all_templates_equal"] = bool(
            np.array_equal(templates["2400"], templates["4800"])
            and np.array_equal(templates["2400"], templates["9600"])
        )
        result[f"Scan{item:05d}"] = item_result
    (WORK_DIR / "csgc_cross_resolution_more_ids.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
