"""CRC-verified HTTP Range extraction of original DIV2K validation images.

Used to establish known digital-source correspondence for the public
DIV2K-SCAN smartphone-photo test archive without waiting for its full source
archive download. Neither photo nor original is assumed pixel-aligned.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import hashlib
import json
import zipfile
from io import BytesIO

from PIL import Image

from palimpsest.io.remote_zip import ChunkedRangeFile, read_member


URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
SIZE = 448_993_893
IDS = ("0801", "0802", "0803")
OUT = DATA_ROOT / "derived/div2k_scan_original_probe"
MANIFEST = WORK_DIR / "div2k_scan_original_range_probe.json"




def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    remote = ChunkedRangeFile(URL, SIZE)
    rows = []
    with zipfile.ZipFile(remote) as archive:
        for image_id in IDS:
            name = f"DIV2K_valid_HR/{image_id}.png"
            member = archive.getinfo(name)
            data = read_member(remote, member)
            with Image.open(BytesIO(data)) as image:
                image.verify()
            with Image.open(BytesIO(data)) as image:
                dimensions = image.size
                mode = image.mode
            target = OUT / f"{image_id}.png"
            target.write_bytes(data)
            rows.append(
                {
                    "id": image_id,
                    "source_member": name,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "dimensions": dimensions,
                    "mode": mode,
                }
            )
    result = {
        "url": URL,
        "http_head_size": SIZE,
        "zip_member_crc_checked": True,
        "range_requests": remote.request_count,
        "range_bytes": remote.bytes_requested,
        "images": rows,
    }
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
