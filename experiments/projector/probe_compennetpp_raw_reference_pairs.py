"""Range/CRC audit of a few same-setup projector and raw-camera pairs.

The ZIP is not downloaded or globally verified. Each selected member's
uncompressed bytes are checked against its ZIP CRC before saving to E:.
"""

from __future__ import annotations

from experiments.paths import WORK_DIR

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.projector.probe_compennetpp_range import SIZE
from experiments.projector.probe_compennetpp_structured_light import confirmed_url
from experiments.print_scan.probe_div2k_scan_originals import ChunkedRangeFile
from experiments.dragotti.probe_dragotti_range import read_member


SETUP = "light1/pos1/cloud_np"
NUMBERS = (1, 2, 3, 31, 63, 95, 125)
TEST_NUMBERS = (1, 2, 3)
DEST = Path(r"E:\ai_image_origin_research\data\derived\compennetpp_raw_ref_probe")
OUT = WORK_DIR / "compennetpp_raw_ref_probe.json"


def image_stats(data: bytes) -> dict:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    return {
        "dimensions": [int(rgb.shape[1]), int(rgb.shape[0])],
        "rgb_mean": rgb.mean((0, 1)).tolist(),
        "rgb_std": rgb.std((0, 1)).tolist(),
        "rgb_corner": rgb[0, 0].tolist(),
        "unique_rgb_count": int(
            np.unique((rgb * 255).astype(np.uint8).reshape(-1, 3), axis=0).shape[0]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", default=SETUP)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--ref-numbers", nargs="*", type=int, default=list(NUMBERS))
    parser.add_argument(
        "--test-numbers", nargs="*", type=int, default=list(TEST_NUMBERS)
    )
    parser.add_argument("--train-numbers", nargs="*", type=int, default=[])
    parser.add_argument("--output-json", type=Path, default=OUT)
    args = parser.parse_args()
    destination = args.destination or (
        DEST
        if args.setup == SETUP
        else DEST.with_name("compennetpp_raw_ref_probe_" + args.setup.replace("/", "_"))
    )
    destination.mkdir(parents=True, exist_ok=True)
    remote = ChunkedRangeFile(confirmed_url(), SIZE)
    rows = []
    counts = {}
    with zipfile.ZipFile(remote) as archive:
        all_names = {info.filename for info in archive.infolist() if not info.is_dir()}
        for category in ("ref", "train", "test"):
            counts[category] = {
                "source": len(
                    [name for name in all_names if name.startswith(f"{category}/")]
                ),
                "raw_camera": len(
                    [
                        name
                        for name in all_names
                        if name.startswith(f"{args.setup}/cam/raw/{category}/")
                    ]
                ),
            }
            numbers = (
                args.ref_numbers
                if category == "ref"
                else args.test_numbers
                if category == "test"
                else args.train_numbers
            )
            for number in numbers:
                for role, name in (
                    ("source", f"{category}/img_{number:04d}.png"),
                    ("camera", f"{args.setup}/cam/raw/{category}/img_{number:04d}.png"),
                ):
                    if name not in all_names:
                        raise RuntimeError(f"missing ZIP member {name}")
                    info = archive.getinfo(name)
                    data = read_member(remote, info)
                    target = destination / name.replace("/", "__")
                    target.write_bytes(data)
                    row = {
                        "name": name,
                        "category": category,
                        "number": number,
                        "role": role,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "zip_crc": f"{info.CRC:08x}",
                        "local": str(target),
                    }
                    row.update(image_stats(data))
                    rows.append(row)
                    print(
                        category,
                        number,
                        role,
                        row["dimensions"],
                        row["rgb_mean"],
                        row["unique_rgb_count"],
                        flush=True,
                    )
    result = {
        "setup": args.setup,
        "archive_bytes": SIZE,
        "full_zip_verified": False,
        "sample_member_crc_verified": True,
        "archive_pair_counts": counts,
        "sample_pairs": rows,
        "range_requests": remote.request_count,
        "range_bytes_requested": remote.bytes_requested,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
