"""Read official CompenNet ZIP directory over verified HTTP Range only.

The full Google Drive archive downloads separately. This helper does not
confuse a remote directory listing with a locally verified dataset.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import hashlib
import html
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from palimpsest.io.remote_zip import RangeFile, read_member


FILE_ID = "1gUTWZLfiGRBgOnZe66ik-m315h4h0gPt"
SIZE = 2_282_667_301
OUTPUT = WORK_DIR / "compennet_remote_zip_probe.json"
SAMPLE_DIR = DATA_ROOT / "derived/compennet_probe"


def _confirmed_url() -> str:
    initial = "https://drive.google.com/uc?" + urllib.parse.urlencode(
        {"export": "download", "id": FILE_ID}
    )
    try:
        with urllib.request.urlopen(initial, timeout=20) as response:
            landing = response.read().decode("utf-8")
    except OSError:
        # The Drive landing endpoint occasionally times out while the signed
        # usercontent URL remains usable. This is a best-effort read-only probe.
        landing = (WORK_DIR / "compennet_drive_landing.html").read_text(
            encoding="utf-8"
        )
    match = re.search(r'name="uuid" value="([^"]+)"', landing)
    if not match:
        raise RuntimeError("official Drive landing did not offer a confirmation UUID")
    return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {
            "id": FILE_ID,
            "export": "download",
            "confirm": "t",
            "uuid": html.unescape(match.group(1)),
        }
    )


def main() -> None:
    remote = RangeFile(_confirmed_url(), SIZE)
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        files = [info for info in infos if not info.is_dir()]
        roots = Counter(info.filename.split("/")[0] for info in files)
        two_levels = Counter("/".join(info.filename.split("/")[:2]) for info in files)
        suffixes = Counter(Path(info.filename).suffix.lower() for info in files)
        selected = []
        # A handful of tiny, distinct-path members proves that the remote
        # central directory refers to actual CRC-valid payloads.
        by_path = sorted(files, key=lambda info: (info.file_size, info.filename))
        seen_second = set()
        for info in by_path:
            key = "/".join(info.filename.split("/")[:3])
            if key in seen_second or info.file_size > 1_000_000:
                continue
            seen_second.add(key)
            selected.append(info)
            if len(selected) >= 8:
                break
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        samples = []
        for index, info in enumerate(selected):
            data = read_member(remote, info)
            destination = (
                SAMPLE_DIR / f"sample_{index:02d}{Path(info.filename).suffix.lower()}"
            )
            destination.write_bytes(data)
            samples.append(
                {
                    "member": info.filename,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "local_probe": str(destination),
                }
            )
        pair_names = [
            "test/img_0001.png",
            "light1/pos1/stripes/cam/warp/test/img_0001.png",
            "light1/pos1/stripes/cam/warp/ref/img_0126.png",
            "train/img_0001.png",
            "light1/pos1/stripes/cam/warp/train/img_0001.png",
        ]
        pair_samples = []
        for name in pair_names:
            data = read_member(remote, archive.getinfo(name))
            destination = SAMPLE_DIR / name.replace("/", "__")
            destination.write_bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                pair_samples.append(
                    {
                        "member": name,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "dimensions": image.size,
                        "mode": image.mode,
                        "metadata_keys": sorted(image.info.keys()),
                        "rgb_mean": array.mean((0, 1)).tolist(),
                        "local_probe": str(destination),
                    }
                )
    result = {
        "source": "official author Google Drive",
        "file_id": FILE_ID,
        "advertised_archive_bytes": SIZE,
        "zip_directory_opened_by_range": True,
        "local_full_archive_verified": False,
        "member_count": len(files),
        "roots": dict(roots),
        "two_levels": dict(two_levels),
        "suffixes": dict(suffixes),
        "first_80_names": [info.filename for info in files[:80]],
        "samples_crc_checked": samples,
        "paired_probe_crc_checked": pair_samples,
        "range_requests": remote.request_count,
        "range_bytes": remote.bytes_requested,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("first_80_names", "two_levels")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
