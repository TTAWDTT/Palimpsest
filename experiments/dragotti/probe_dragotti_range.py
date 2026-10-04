"""Extract a few verified members from Dragotti's public ZIP via HTTP Range.

This is an audit helper, not a downloader for the full dataset. Each requested
member is checked against the ZIP central-directory size/CRC and its SHA-256
is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path


BASE_URL = "https://www.commsp.ee.ic.ac.uk/~pld/research/Rewind/Recapture/TestImages/"
ARCHIVES = {
    "recaptured": (BASE_URL + "RecapturedImages.zip", 5_224_454_019),
    "original": (BASE_URL + "SingleCaptureImages.zip", 3_213_198_364),
}


def read_member(remote: "RangeFile", info: zipfile.ZipInfo) -> bytes:
    """Fetch one member payload in one Range request after checking its header."""
    if info.flag_bits & 1 or info.compress_type not in (
        zipfile.ZIP_STORED,
        zipfile.ZIP_DEFLATED,
    ):
        raise RuntimeError(f"unsupported ZIP member: {info.filename}")
    remote.seek(info.header_offset)
    header = remote.read(30)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"invalid local header: {info.filename}")
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    remote.seek(info.header_offset + 30)
    name_bytes = remote.read(name_size)
    if (
        name_bytes.decode("utf-8" if info.flag_bits & 0x800 else "cp437")
        != info.filename
    ):
        raise RuntimeError(f"local/central names disagree: {info.filename}")
    remote.seek(info.header_offset + 30 + name_size + extra_size)
    compressed = remote.read(info.compress_size)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    else:
        data = compressed
    if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise RuntimeError(f"size/CRC mismatch: {info.filename}")
    return data


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.pos = 0
        self.request_count = 0
        self.bytes_requested = 0
        self.cache_start = -1
        self.cache = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self.pos + offset
        elif whence == io.SEEK_END:
            pos = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if pos < 0:
            raise ValueError("negative position")
        self.pos = pos
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n == 0:
            return b""
        start = self.pos
        if self.cache_start <= start and start + n <= self.cache_start + len(
            self.cache
        ):
            offset = start - self.cache_start
            self.pos += n
            return self.cache[offset : offset + n]
        # zipfile often reads a compressed member in 4 KiB chunks. Cache a
        # contiguous MiB to avoid one HTTP request per tiny read.
        fetch_size = min(max(n, 1_048_576), self.size - start)
        end = start + fetch_size - 1
        req = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "research-zip-audit/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            content_range = response.headers.get("Content-Range", "")
            expected_range = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or content_range != expected_range:
                raise OSError(
                    f"unexpected HTTP range: {response.status} {content_range}"
                )
            data = response.read()
        if len(data) != fetch_size:
            raise OSError(f"short range read: {len(data)} != {fetch_size}")
        self.cache_start = start
        self.cache = data
        self.pos += n
        self.request_count += 1
        self.bytes_requested += fetch_size
        return data[:n]


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
            selected = lambda info: info.filename.endswith(f"%{args.key}.png") and (
                not args.cameras or info.filename.split("/")[1] in args.cameras
            )
        else:
            selected = (
                lambda info: info.filename.endswith(
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
