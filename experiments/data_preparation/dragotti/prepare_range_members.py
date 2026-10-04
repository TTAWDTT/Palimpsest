"""Extract a few verified members from Dragotti's public ZIP via HTTP Range.

This is an audit helper, not a downloader for the full dataset. Each requested
member is checked against the ZIP central-directory size/CRC and its SHA-256
is recorded.
"""

from __future__ import annotations
from palimpsest.io.remote_zip import RangeFile, read_member

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


BASE_URL = "https://www.commsp.ee.ic.ac.uk/~pld/research/Rewind/Recapture/TestImages/"
ARCHIVES = {
    "recaptured": (BASE_URL + "RecapturedImages.zip", 5_224_454_019),
    "original": (BASE_URL + "SingleCaptureImages.zip", 3_213_198_364),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", default="D40-015", help="Original camera and image number")
    p.add_argument("--archive", choices=ARCHIVES, default="recaptured")
    p.add_argument(
        "--cameras", nargs="*", help="Recapture camera directory names to extract"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--list-only", action="store_true")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9]+-[0-9]{3}", args.key):
        p.error("key must look like D40-015")
    url, expected_size = ARCHIVES[args.archive]

    remote = RangeFile(url, expected_size)
    with zipfile.ZipFile(remote) as archive:
        original_camera, image_number = args.key.split("-")
        if args.archive == "recaptured":
            selected = lambda info: (
                info.filename.endswith(f"%{args.key}.png")
                and (not args.cameras or info.filename.split("/")[1] in args.cameras)
            )
        else:
            selected = lambda info: (
                info.filename.endswith(
                    f"-{int(image_number):04d}-S%{original_camera}.JPG"
                )
                and "not used" not in info.filename.lower()
            )
        members = sorted(
            (info for info in archive.infolist() if selected(info)),
            key=lambda info: info.filename,
        )
        if not members:
            raise RuntimeError(f"no members for {args.key}")
        print(
            json.dumps(
                [
                    {"name": x.filename, "size": x.file_size, "crc32": f"{x.CRC:08x}"}
                    for x in members
                ],
                indent=2,
            )
        )
        if args.list_only:
            return

        args.output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for info in members:
            # Restrict output to flat basenames; never trust member paths.
            if info.is_dir() or info.file_size > 20_000_000:
                raise RuntimeError(f"unexpected member: {info.filename}")
            name = Path(info.filename).name
            target = args.output_dir / name
            if target.exists():
                data = target.read_bytes()
            else:
                data = read_member(remote, info)
                target.write_bytes(data)
            if (
                len(data) != info.file_size
                or (zipfile.crc32(data) & 0xFFFFFFFF) != info.CRC
            ):
                raise RuntimeError(f"invalid local file: {target}")
            record = {
                "zip_member": info.filename,
                "file": str(target),
                "bytes": len(data),
                "crc32": f"{info.CRC:08x}",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "archive_url": url,
                    "archive_bytes": expected_size,
                    "source_key": args.key,
                    "records": records,
                    "range_requests": remote.request_count,
                    "range_bytes": remote.bytes_requested,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
