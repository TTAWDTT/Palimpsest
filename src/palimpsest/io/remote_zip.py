"""Seekable HTTP Range access and CRC-checked ZIP member reading."""

from __future__ import annotations

import io
import http.client
import time
import struct
import urllib.request
import zipfile
import zlib


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
        if n <= 0:
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


class ChunkedRangeFile(RangeFile):
    """Bound each request; the ETH server dropped a multi-MB Range response."""

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        pieces = []
        remaining = n
        while remaining:
            count = min(remaining, 262_144)
            start, end = self.pos, self.pos + count - 1
            for attempt in range(5):
                try:
                    request = urllib.request.Request(
                        self.url, headers={"Range": f"bytes={start}-{end}"}
                    )
                    with urllib.request.urlopen(request, timeout=60) as response:
                        expected = f"bytes {start}-{end}/{self.size}"
                        if (
                            response.status != 206
                            or response.headers.get("Content-Range") != expected
                        ):
                            raise OSError("invalid HTTP Range response")
                        payload = response.read()
                    if len(payload) != count:
                        raise OSError("short HTTP Range response")
                    break
                except (OSError, ValueError, http.client.IncompleteRead):
                    if attempt == 4:
                        raise
                    time.sleep(attempt + 1)
            pieces.append(payload)
            self.pos += count
            self.request_count += 1
            self.bytes_requested += count
            remaining -= count
        return b"".join(pieces)
