"""Fetch a bounded set of official ImageNet-ES same-source aperture pairs."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import argparse
import hashlib
import io
import json
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import requests
from PIL import ExifTags, Image

from experiments.imagenet_es.audit_imagenet_es_remote_index import (
    RequestsRangeFile,
    URL,
)
from experiments.dragotti.probe_dragotti_range import read_member


ROOT = DATA_ROOT / "derived/imagenet_es_probe"
OUT = WORK_DIR / "imagenet_es_control_probe_manifest.json"
PARAM_IDS = (4, 5, 13, 14, 22, 23)  # ISO 250/2000, 1/60 s, f/5/9/16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--class-offset", type=int, default=0)
    parser.add_argument("--params", type=int, nargs="+", default=PARAM_IDS)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if (
        not 1 <= args.classes <= 200
        or args.class_offset < 0
        or args.class_offset + args.classes > 200
        or any(not 1 <= p <= 27 for p in args.params)
    ):
        parser.error("class offset/count must select 1..200 and param IDs 1..27")
    head = requests.head(URL, allow_redirects=True, timeout=30)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    remote = RequestsRangeFile(head.url, size)
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(remote) as archive:
        info_by_name = {
            info.filename: info for info in archive.infolist() if not info.is_dir()
        }
        prefix = "ImageNet-ES/es-test/sampled_tin_no_resize2/"
        by_class = defaultdict(list)
        for name in info_by_name:
            if name.startswith(prefix) and name.lower().endswith(".jpeg"):
                by_class[name.split("/")[3]].append(name)
        classes = sorted(
            by_class, key=lambda name: hashlib.sha256(name.encode()).digest()
        )[args.class_offset : args.class_offset + args.classes]
        selected = []
        for cls in classes:
            ref = sorted(
                by_class[cls], key=lambda name: hashlib.sha256(name.encode()).digest()
            )[0]
            selected.append(ref)
            stem = PurePosixPath(ref).name
            for param in args.params:
                capture = (
                    f"ImageNet-ES/es-test/param_control/l5/param_{param}/{cls}/{stem}"
                )
                if capture not in info_by_name:
                    raise RuntimeError(f"missing official pair {capture}")
                selected.append(capture)
        rows = []
        reused = 0
        for name in selected:
            info = info_by_name[name]
            parts = PurePosixPath(name).parts
            destination = ROOT.joinpath(*parts[1:])
            if not destination.resolve().is_relative_to(ROOT.resolve()):
                raise RuntimeError("archive member escapes output root")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                data = destination.read_bytes()
                if (
                    len(data) == info.file_size
                    and zlib.crc32(data) & 0xFFFFFFFF == info.CRC
                ):
                    reused += 1
                else:
                    data = read_member(remote, info)
                    destination.write_bytes(data)
            else:
                data = read_member(remote, info)
                destination.write_bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                exif = {
                    ExifTags.TAGS.get(key, str(key)): str(value)
                    for key, value in image.getexif().items()
                }
                rows.append(
                    {
                        "member": name,
                        "local_path": str(destination),
                        "bytes": len(data),
                        "crc32": f"{info.CRC:08x}",
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                        "rgb_mean": rgb.mean((0, 1)).tolist(),
                        "fraction_rgb_near_black": float(
                            (rgb.max(axis=2) <= 0.02).mean()
                        ),
                        "fraction_rgb_near_white": float(
                            (rgb.min(axis=2) >= 0.98).mean()
                        ),
                        "exif": exif,
                    }
                )
    report = {
        "url": URL,
        "archive_bytes": size,
        "etag": head.headers.get("ETag"),
        "source_classes": classes,
        "class_offset": args.class_offset,
        "param_ids": args.params,
        "mapping_source": "author repository settings/grid-options-3x3x3.csv",
        "rows": rows,
        "reused_verified_local_files": reused,
        "range_requests": remote.request_count,
        "range_bytes": remote.bytes_requested,
        "qualification": "JPEG cropped release, not original RAW; same source and controlled nominal settings but actual screen/camera spectral geometry unavailable",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_classes": classes,
                "files": len(rows),
                "reused": reused,
                "range_requests": remote.request_count,
                "range_bytes": remote.bytes_requested,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
