"""Inspect author CompenNet++ ZIP central directory over HTTP Range only."""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import html
import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from collections import Counter

import numpy as np
from PIL import Image

from experiments.print_scan.probe_div2k_scan_originals import ChunkedRangeFile
from experiments.dragotti.probe_dragotti_range import read_member


FILE_ID = "1XK6jzNvLwbc-XFg3IU_KpJMT4yB_3JVS"
SIZE = 11_476_313_823
OUTPUT = WORK_DIR / "compennetpp_remote_zip_probe.json"


def main() -> None:
    landing_url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
        {"export": "download", "id": FILE_ID}
    )
    try:
        with urllib.request.urlopen(landing_url, timeout=20) as response:
            landing = response.read().decode("utf-8")
    except OSError:
        landing = (WORK_DIR / "compennetpp_drive_landing.html").read_text(
            encoding="utf-8"
        )
    match = re.search(r'name="uuid" value="([^"]+)"', landing)
    if not match:
        raise RuntimeError("no public Drive confirmation token")
    url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {
            "id": FILE_ID,
            "export": "download",
            "confirm": "t",
            "uuid": html.unescape(match.group(1)),
        }
    )
    remote = ChunkedRangeFile(url, SIZE)
    with zipfile.ZipFile(remote) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        sample_names = [
            "README.md",
            "test/img_0001.png",
            "light1/pos1/cloud_np/cam/raw/test/img_0001.png",
            "light1/pos1/cloud_np/cam/raw/ref/img_0126.png",
        ]
        sample_dir = DATA_ROOT / "derived/compennetpp_probe"
        sample_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for name in sample_names:
            info = archive.getinfo(name)
            data = read_member(remote, info)
            destination = sample_dir / name.replace("/", "__")
            destination.write_bytes(data)
            row = {
                "member": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "local_probe": str(destination),
            }
            if name.lower().endswith(".png"):
                with Image.open(io.BytesIO(data)) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                    row.update(
                        {
                            "dimensions": image.size,
                            "mode": image.mode,
                            "metadata_keys": sorted(image.info.keys()),
                            "rgb_mean": array.mean((0, 1)).tolist(),
                        }
                    )
            else:
                row["text"] = data.decode("utf-8", errors="replace")
            samples.append(row)
    roots = Counter(name.split("/")[0] for name in names)
    markers = {
        term: sum(term in name.lower() for name in names)
        for term in ("raw", "warp", "train", "test", "ref", "surface")
    }
    examples = {
        term: [name for name in names if term in name.lower()][:15]
        for term in ("raw", "warp", "train", "test", "ref")
    }
    result = {
        "source": "official author Google Drive",
        "archive_bytes": SIZE,
        "remote_directory_only": True,
        "member_count": len(names),
        "roots": dict(roots),
        "markers": markers,
        "examples": examples,
        "samples_crc_checked": samples,
        "range_requests": remote.request_count,
        "range_bytes": remote.bytes_requested,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
