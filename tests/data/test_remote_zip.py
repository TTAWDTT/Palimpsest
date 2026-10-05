import io
import zipfile
from types import SimpleNamespace

import pytest

from palimpsest.io.remote_zip import ChunkedRangeFile, RangeFile, read_member


def test_range_reader_works_with_zipfile_and_reuses_cache(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("folder/图.txt", "verified payload".encode())
    payload = buffer.getvalue()
    requests = []

    class Response(io.BytesIO):
        status = 206

        def __init__(self, request):
            begin, end = map(int, request.get_header("Range")[6:].split("-"))
            super().__init__(payload[begin : end + 1])
            self.headers = {"Content-Range": f"bytes {begin}-{end}/{len(payload)}"}
            requests.append((begin, end))

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout: Response(request)
    )
    remote = RangeFile("https://example.invalid/archive.zip", len(payload))
    with zipfile.ZipFile(remote) as archive:
        assert (
            read_member(remote, archive.getinfo("folder/图.txt")) == b"verified payload"
        )
    before = len(requests)
    remote.seek(0)
    assert remote.read(4) == b"PK\x03\x04"
    remote.seek(0)
    assert remote.read(4) == b"PK\x03\x04"
    assert len(requests) <= before + 1
    remote.seek(len(payload) + 10)
    assert remote.read(10) == b""


def test_range_reader_rejects_ignored_range(monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout: Response(b"archive")
    )
    with pytest.raises(OSError, match="unexpected HTTP range"):
        RangeFile("https://example.invalid/zip", 7).read(1)


def test_member_checks_crc_and_rejects_encryption():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("source.bin", b"original")
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo("source.bin")
        damaged = payload.replace(b"original", b"tampered", 1)
        with pytest.raises(RuntimeError, match="size/CRC mismatch"):
            read_member(io.BytesIO(damaged), info)
    encrypted = SimpleNamespace(flag_bits=1, filename="secret.bin")
    with pytest.raises(RuntimeError, match="unsupported ZIP member"):
        read_member(io.BytesIO(), encrypted)


def test_bounded_reader_splits_requests_and_retries_short_responses(monkeypatch):
    payload = b"x" * (262_144 + 17)
    spans = []
    first = True

    class Response(io.BytesIO):
        status = 206

        def __init__(self, request):
            nonlocal first
            begin, end = map(int, request.get_header("Range")[6:].split("-"))
            data = payload[begin : end + 1]
            if first:
                first = False
                data = data[:-1]
            super().__init__(data)
            self.headers = {"Content-Range": f"bytes {begin}-{end}/{len(payload)}"}
            spans.append((begin, end))

    sleeps = []
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout: Response(request)
    )
    monkeypatch.setattr("time.sleep", sleeps.append)
    remote = ChunkedRangeFile("https://example.invalid/zip", len(payload))
    assert remote.read() == payload
    assert spans == [(0, 262_143), (0, 262_143), (262_144, 262_160)]
    assert sleeps == [1]
    assert remote.request_count == 2 and remote.bytes_requested == len(payload)
    remote.seek(len(payload) + 1)
    assert remote.read(10) == b""
