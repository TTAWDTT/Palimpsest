"""CRC-probe structured-light inputs and unwarped captures without 11 GB ZIP.

This is a Range-only sample, never a full-archive verification. The author
labels `cam/raw` as unwarped camera RGB PNG, not sensor RAW.
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

import numpy as np
from PIL import Image

from experiments.projector.probe_compennetpp_range import FILE_ID, SIZE
from experiments.print_scan.probe_div2k_scan_originals import ChunkedRangeFile
from experiments.dragotti.probe_dragotti_range import read_member


OUT = WORK_DIR / "compennetpp_structured_light_probe.json"
SAMPLE_DIR = DATA_ROOT / "derived/compennetpp_sl_probe"


def confirmed_url() -> str:
    initial = "https://drive.google.com/uc?" + urllib.parse.urlencode(
        {"export": "download", "id": FILE_ID}
    )
    try:
        with urllib.request.urlopen(initial, timeout=20) as response:
            landing = response.read().decode("utf-8")
    except OSError:
        landing = (WORK_DIR / "compennetpp_drive_landing.html").read_text(
            encoding="utf-8"
        )
    match = re.search(r'name="uuid" value="([^"]+)"', landing)
    if not match:
        raise RuntimeError("official Drive landing has no public confirmation UUID")
    return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {
            "id": FILE_ID,
            "export": "download",
            "confirm": "t",
            "uuid": html.unescape(match.group(1)),
        }
    )


def probe(data: bytes, name: str) -> dict:
    with Image.open(io.BytesIO(data)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        return {
            "member": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "dimensions": image.size,
            "mode": image.mode,
            "metadata_keys": sorted(image.info),
            "rgb_mean": array.mean((0, 1)).tolist(),
            "rgb_std": array.std((0, 1)).tolist(),
        }


def main() -> None:
    remote = ChunkedRangeFile(confirmed_url(), SIZE)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(remote) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        root_sl = sorted(name for name in names if name.startswith("sl/"))
        capture_sl = [name for name in names if "/cam/raw/sl/" in name]
        setup_counts = Counter(name.split("/cam/raw/sl/")[0] for name in capture_sl)
        sample_setups = ["light1/pos1/cloud_np", "light2/pos1/cloud_np"]
        sample_numbers = [1, 2, 21, 42]
        sample_names = []
        for number in sample_numbers:
            source = f"sl/img_{number:04d}.png"
            if source in names:
                sample_names.append(source)
            for setup in sample_setups:
                capture = f"{setup}/cam/raw/sl/img_{number:04d}.png"
                if capture in names:
                    sample_names.append(capture)
        samples = []
        for name in sample_names:
            data = read_member(remote, archive.getinfo(name))
            destination = SAMPLE_DIR / name.replace("/", "__")
            destination.write_bytes(data)
            row = probe(data, name)
            row["local_probe"] = str(destination)
            samples.append(row)
            print(name, row["dimensions"], row["sha256"][:12], flush=True)
    result = {
        "official_file_id": FILE_ID,
        "archive_bytes": SIZE,
        "remote_directory_only": True,
        "full_zip_crc_checked": False,
        "root_sl_count": len(root_sl),
        "root_sl_names": root_sl,
        "raw_sl_setup_counts": dict(sorted(setup_counts.items())),
        "samples_crc_checked": samples,
        "range_requests": remote.request_count,
        "range_bytes_requested": remote.bytes_requested,
    }
    OUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("setups", len(setup_counts), "sample images", len(samples), flush=True)


if __name__ == "__main__":
    main()
