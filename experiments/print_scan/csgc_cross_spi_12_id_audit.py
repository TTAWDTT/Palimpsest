"""Range audit 12 predetermined CSGC holdout IDs across 4800/9600 SPI.

The 2400 archive is already fully CRC-audited. Every new ZIP member is
individually length/CRC-checked by `extract`; the binary template must match
the canonical 2400 archive. This audits digital identity, not physical paper.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image

from experiments.print_scan.probe_csgc_cross_resolution import extract


ARCHIVE = DATA_ROOT / "raw/csgc/CSGC_scan2400spi.zip"
OUT = WORK_DIR / "csgc_cross_spi_12_id_audit.json"
IDS = (125, 200, 275, 350, 425, 500, 575, 650, 725, 800, 875, 950)

rows = []
with zipfile.ZipFile(ARCHIVE) as z:
    for item in IDS:
        name = f"Scan{item:05d}.tif"
        with Image.open(
            io.BytesIO(z.read(f"2400dpi_NEW/bin_exact_crop_4/{name}"))
        ) as im:
            source = np.asarray(im, dtype=np.uint8)[::4, ::4]
        source_hash = hashlib.sha256(source.tobytes()).hexdigest()
        for spi, factor in (("4800", 8), ("9600", 16)):
            binary_path, binary_record = extract(spi, f"bin_exact_crop_{factor}", item)
            _, scan_record = extract(spi, "resize_exact_crop", item)
            with Image.open(binary_path) as im:
                binary = np.asarray(im, dtype=np.uint8)
            if binary.shape != (100 * factor, 100 * factor):
                raise RuntimeError(f"{spi}/{name}: wrong binary shape")
            if not np.all(
                binary.reshape(100, factor, 100, factor).max(axis=(1, 3))
                == binary.reshape(100, factor, 100, factor).min(axis=(1, 3))
            ):
                raise RuntimeError(f"{spi}/{name}: nonuniform binary blocks")
            match = bool(np.array_equal(binary[::factor, ::factor], source))
            if not match:
                raise RuntimeError(f"{spi}/{name}: digital templates do not match")
            rows.append(
                {
                    "id": item,
                    "spi": int(spi),
                    "source_sha256": source_hash,
                    "binary_sha256": binary_record["sha256"],
                    "scan_sha256": scan_record["sha256"],
                    "template_equal_2400": match,
                }
            )
            print(f"verified {spi} {name}", flush=True)

report = {
    "ids_preselected": list(IDS),
    "n_verified_members": len(rows) * 2,
    "n_verified_cross_spi_pairs": len(rows),
    "paper_identity_verified": False,
    "rows": rows,
}
OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            k: report[k]
            for k in (
                "ids_preselected",
                "n_verified_members",
                "n_verified_cross_spi_pairs",
            )
        },
        indent=2,
    )
)
