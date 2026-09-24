"""Extract 42 structured-light source/camera pairs via CRC-verified ZIP Range.

Only one CompenNet++ setup is selected, requiring a small part of the public
11.5 GB archive. It is resumable and makes no whole-archive integrity claim.
All image payloads remain on E:; the manifest is in work/.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zlib
import zipfile
from pathlib import Path

from PIL import Image

from work.probe_compennetpp_range import SIZE
from work.probe_compennetpp_structured_light import confirmed_url
from work.probe_div2k_scan_originals import ChunkedRangeFile
from work.probe_dragotti_range import read_member


SETUP = "light1/pos1/cloud_np"
OUT = Path(__file__).with_name("compennetpp_sl_one_setup_manifest.json")
FOLDER = Path(r"E:\ai_image_origin_research\data\derived\compennetpp_sl_one_setup")


def valid_local(path: Path, info: zipfile.ZipInfo) -> bool:
    if not path.exists() or path.stat().st_size != info.file_size:
        return False
    data = path.read_bytes()
    return (zlib.crc32(data) & 0xFFFFFFFF) == info.CRC


def main() -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    remote = ChunkedRangeFile(confirmed_url(), SIZE)
    rows = []
    started = time.perf_counter()
    with zipfile.ZipFile(remote) as archive:
        expected = [(kind, index,
                     f"sl/img_{index:04d}.png" if kind == "source" else
                     f"{SETUP}/cam/raw/sl/img_{index:04d}.png")
                    for index in range(1, 43) for kind in ("source", "capture")]
        for kind, index, member in expected:
            info = archive.getinfo(member)
            destination = FOLDER / f"{kind}_{index:04d}.png"
            if not valid_local(destination, info):
                data = read_member(remote, info)
                temp = destination.with_suffix(".png.tmp")
                temp.write_bytes(data)
                temp.replace(destination)
            data = destination.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                row = {"kind": kind, "index": index, "member": member,
                       "file_size": info.file_size, "zip_crc32": f"{info.CRC:08x}",
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "size": image.size, "mode": image.mode,
                       "metadata_keys": sorted(image.info),
                       "local_path": str(destination)}
            rows.append(row)
            if index % 4 == 0 and kind == "capture":
                print(f"CRC checked {len(rows)}/84 members; elapsed {time.perf_counter()-started:.0f}s",
                      flush=True)
            OUT.write_text(json.dumps({"setup": SETUP, "archive_bytes": SIZE,
                                       "whole_archive_crc_checked": False,
                                       "pairs_complete": len(rows) == 84,
                                       "range_requests": remote.request_count,
                                       "range_bytes_requested": remote.bytes_requested,
                                       "elapsed_sec": time.perf_counter() - started,
                                       "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print("done", len(rows), "members", flush=True)


if __name__ == "__main__":
    main()
