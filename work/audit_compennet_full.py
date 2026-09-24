"""Audit the complete official CompenNet projector-camera benchmark archive.

ZIP CRC is checked for every member. Train/test source IDs and every setup's
captured outputs are inventoried without treating aligned 256-pixel PNGs as
native camera frames. Raw archive remains on E:.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ARCHIVE = Path(r"E:\ai_image_origin_research\data\raw\compennet\CompenNetDataset.zip")
OUTPUT = Path(__file__).with_name("compennet_full_audit.json")
EXPECTED_BYTES = 2_282_667_301


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_probe(data: bytes) -> dict:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        return {"size": image.size, "mode": image.mode,
                "metadata_keys": sorted(image.info.keys()),
                "rgb_mean": array.mean((0, 1)).tolist()}


def main() -> None:
    if ARCHIVE.stat().st_size != EXPECTED_BYTES:
        raise ValueError("official archive not complete at expected byte count")
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if archive.testzip() is not None:
            raise ValueError("at least one member failed ZIP CRC")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member name")
        train = sorted(name for name in names if name.startswith("train/") and name.endswith(".png"))
        test = sorted(name for name in names if name.startswith("test/") and name.endswith(".png"))
        ref = sorted(name for name in names if name.startswith("ref/") and name.endswith(".png"))
        if train != [f"train/img_{i:04d}.png" for i in range(1, 501)]:
            raise ValueError("input training IDs differ from 1-500")
        if test != [f"test/img_{i:04d}.png" for i in range(1, 201)]:
            raise ValueError("input test IDs differ from 1-200")
        # The 125 uniform calibration colors form a 5^3 RGB grid; the
        # additional gray source has a distinct filename. Captured outputs
        # use numeric 0001..0126, so the two namespaces must not be conflated.
        expected_ref = sorted([f"ref/img_{i:04d}.png" for i in range(1, 126)]
                              + ["ref/img_gray.png"])
        if ref != expected_ref:
            raise ValueError("input reference IDs differ from 125-grid plus gray")
        setups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for name in names:
            parts = name.split("/")
            if len(parts) < 4 or not parts[0].startswith("light"):
                continue
            key = "/".join(parts[:3])
            category = "/".join(parts[3:-1])
            setups[key][category].append(name)
        summary = {}
        for key, groups in sorted(setups.items()):
            warped_train = sorted(groups.get("cam/warp/train", []))
            warped_test = sorted(groups.get("cam/warp/test", []))
            warped_ref = sorted(groups.get("cam/warp/ref", []))
            summary[key] = {"member_count": sum(len(v) for v in groups.values()),
                            "categories": {category: len(values) for category, values in sorted(groups.items())},
                            "warped_train_complete": warped_train ==
                                [f"{key}/cam/warp/train/img_{i:04d}.png" for i in range(1, 501)],
                            "warped_test_complete": warped_test ==
                                [f"{key}/cam/warp/test/img_{i:04d}.png" for i in range(1, 201)],
                            "warped_ref_complete": warped_ref ==
                                [f"{key}/cam/warp/ref/img_{i:04d}.png" for i in range(1, 127)]}
        sample_names = ["test/img_0001.png", "train/img_0001.png", "ref/img_gray.png"]
        for key in sorted(setups):
            sample_names.extend([f"{key}/cam/warp/test/img_0001.png",
                                 f"{key}/cam/warp/ref/img_0126.png"])
        sample_probes = {name: _image_probe(archive.read(name)) for name in sample_names}
        captured_126_duplicates = {}
        first_125_unique = {}
        for key in sorted(setups):
            prefix = f"{key}/cam/warp/ref/"
            last_digest = hashlib.sha256(archive.read(prefix + "img_0126.png")).digest()
            first_digests = [hashlib.sha256(archive.read(prefix + f"img_{index:04d}.png")).digest()
                             for index in range(1, 126)]
            matches = [index for index, digest in enumerate(first_digests, 1)
                       if digest == last_digest]
            captured_126_duplicates[key] = matches
            first_125_unique[key] = len(set(first_digests)) == 125
        readme_files = {name: archive.read(name).decode("utf-8", errors="replace")
                        for name in names if name.lower().endswith(".txt")}
    result = {"archive": str(ARCHIVE), "archive_bytes": EXPECTED_BYTES,
              "archive_sha256": _sha256(ARCHIVE), "all_member_crc_passed": True,
              "file_count": len(infos), "root_counts": dict(Counter(name.split("/")[0] for name in names)),
              "setup_count": len(setups), "setups": summary,
              "captured_ref_0126_exact_duplicate_of": captured_126_duplicates,
              "captured_first_125_references_unique": first_125_unique,
              "sample_probes": sample_probes, "readme_files": readme_files,
              "capture_status": "released warped/registered 256px PNG; native camera frames not established"}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("setups", "sample_probes", "readme_files", "captured_ref_0126_exact_duplicate_of", "captured_first_125_references_unique")},
                     ensure_ascii=False, indent=2))
    print("reference 0126 duplicate IDs", dict(Counter(
        tuple(matches) for matches in captured_126_duplicates.values())))
    print("first 125 unique in", sum(first_125_unique.values()), "setups")
    print("complete setups", sum(row["warped_train_complete"] and row["warped_test_complete"]
                                  and row["warped_ref_complete"] for row in summary.values()),
          "/", len(summary))


if __name__ == "__main__":
    main()
