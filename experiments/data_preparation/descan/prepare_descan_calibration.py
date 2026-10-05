"""Download and verify the public paired DESCAN-18K validation/test archives."""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

from palimpsest.paths import DATA_ROOT

from palimpsest.io.hashing import file_sha256 as hash_file


import json
import stat
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


REPO = "ENCLab/DESCAN-18K"
REVISION = "b5617e5ef9217116daa0ed2740394cc1ef2e03d7"
DESTINATION = DATA_ROOT / "raw/descan18k"
EXPECTED = {
    "Valid.zip": (
        754770440,
        "f515fc2723db71e709588b2711577085318611684978fe4ad562042f452d6b2b",
    ),
    "Test.zip": (
        767709023,
        "93c904410f409c1a393e5e79b35bf1748cb5d0ad2d18d1e1958d32213a9d8304",
    ),
}
SUMMARY = WORK_DIR / "descan_archive_audit.json"


class ResumeNotSupported(Exception):
    """The server did not return the requested byte range."""


def audit_zip(path: Path) -> dict[str, int]:
    seen = set()
    image_count = 0
    uncompressed_bytes = 0
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            name = PurePosixPath(normalized_name)
            windows_name = PureWindowsPath(member.filename)
            if (
                name.is_absolute()
                or windows_name.drive
                or windows_name.is_absolute()
                or ".." in name.parts
                or not name.parts
            ):
                raise ValueError(f"unsafe path in {path.name}: {member.filename}")
            canonical_name = name.as_posix().casefold()
            if canonical_name in seen:
                raise ValueError(f"duplicate member in {path.name}: {member.filename}")
            seen.add(canonical_name)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink in {path.name}: {member.filename}")
            uncompressed_bytes += member.file_size
            if name.suffix.lower() in {".tif", ".tiff"}:
                image_count += 1
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise ValueError(f"corrupt ZIP member in {path.name}: {corrupt_member}")
    return {
        "members": len(seen),
        "tiff_images": image_count,
        "uncompressed_bytes": uncompressed_bytes,
    }


def download_archive(filename: str, expected_size: int) -> Path:
    final_path = DESTINATION / filename
    expected_sha256 = EXPECTED[filename][1]
    if final_path.exists():
        if (
            final_path.stat().st_size == expected_size
            and hash_file(final_path) == expected_sha256
        ):
            return final_path
        final_path.unlink()
    partial_path = DESTINATION / f"{filename}.partial"
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}?download=true"
    for attempt in range(8):
        offset = partial_path.stat().st_size if partial_path.exists() else 0
        if offset == expected_size:
            if hash_file(partial_path) == expected_sha256:
                partial_path.replace(final_path)
                return final_path
            partial_path.unlink()
            offset = 0
        if offset > expected_size:
            partial_path.unlink()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.status
                if offset and status != 206:
                    raise ResumeNotSupported(
                        f"server ignored Range for {filename} at byte {offset}"
                    )
                if offset and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ResumeNotSupported(
                        f"invalid Content-Range for {filename} at byte {offset}"
                    )
                if not offset and status != 200:
                    raise ValueError(f"unexpected HTTP {status} for {filename}")
                with partial_path.open("ab") as output:
                    next_report = (offset // (64 * 1024 * 1024) + 1) * (
                        64 * 1024 * 1024
                    )
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        if output.tell() >= next_report:
                            print(
                                f"{filename}: {output.tell()}/{expected_size}",
                                flush=True,
                            )
                            next_report += 64 * 1024 * 1024
        except ResumeNotSupported as error:
            print(f"{filename}: restarting from byte zero: {error}", flush=True)
            partial_path.unlink(missing_ok=True)
            continue
        except (OSError, TimeoutError) as error:
            print(f"{filename}: retry {attempt + 1}: {error}", flush=True)
            time.sleep(min(2**attempt, 30))
            continue
        if partial_path.stat().st_size == expected_size:
            if hash_file(partial_path) == expected_sha256:
                partial_path.replace(final_path)
                return final_path
            partial_path.unlink()
            print(f"{filename}: SHA-256 mismatch; retrying from byte zero", flush=True)
            continue
        if partial_path.stat().st_size > expected_size:
            raise ValueError(f"download exceeded expected size: {filename}")
    raise RuntimeError(f"could not complete {filename} after eight attempts")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records = {}
    for filename, (expected_size, expected_sha256) in EXPECTED.items():
        path = download_archive(filename, expected_size)
        if path.stat().st_size != expected_size:
            raise ValueError(f"size mismatch: {filename}")
        actual_sha256 = hash_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"SHA-256 mismatch: {filename}")
        archive_audit = audit_zip(path)
        records[filename] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
            **archive_audit,
        }
        print(json.dumps({filename: records[filename]}, ensure_ascii=False), flush=True)
    SUMMARY.write_text(
        json.dumps(
            {"repository": REPO, "revision": REVISION, "files": records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
