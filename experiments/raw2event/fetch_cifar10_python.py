"""Fetch an official-MD5 CIFAR-10 archive mirror for exact stimulus lookup."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import requests


URL = "https://huggingface.co/datasets/VerisimilitudeX/cifar10/resolve/main/cifar-10-python.tar.gz"
OFFICIAL = "https://www.cs.toronto.edu/~kriz/cifar.html"
ROOT = Path("E:/ai_image_origin_research/data/raw/cifar10_official")
TARGET = ROOT / "cifar-10-python.tar.gz"
AUDIT = Path("work/cifar10_python_download_audit.json")
EXPECTED_LENGTH = 170_498_071
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fetch_range(partial: Path, start: int, stop: int) -> None:
    for attempt in range(5):
        try:
            with requests.get(
                URL,
                headers={"Range": f"bytes={start}-{stop}"},
                stream=True,
                timeout=(30, 60),
            ) as response:
                response.raise_for_status()
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {start}-{stop}/{EXPECTED_LENGTH}"
                ):
                    raise RuntimeError(f"mirror ignored range {start}-{stop}")
                written = 0
                with partial.open("r+b") as output:
                    output.seek(start)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                            written += len(chunk)
                if written != stop - start + 1:
                    raise RuntimeError(f"incomplete range {start}-{stop}: {written}")
            return
        except (requests.RequestException, RuntimeError):
            if attempt == 4:
                raise
            time.sleep(1 + attempt)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    reused = TARGET.exists()
    if not reused:
        partial = TARGET.with_suffix(TARGET.suffix + ".partial")
        # Fixed, non-recursive temporary file in the data directory.
        with partial.open("wb") as stream:
            stream.truncate(EXPECTED_LENGTH)
        partitions = []
        width = 4 * 1024 * 1024
        for start in range(0, EXPECTED_LENGTH, width):
            partitions.append((start, min(EXPECTED_LENGTH - 1, start + width - 1)))
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(fetch_range, partial, start, stop)
                for start, stop in partitions
            ]
            for future in futures:
                future.result()
        if partial.stat().st_size != EXPECTED_LENGTH or md5(partial) != EXPECTED_MD5:
            raise RuntimeError("CIFAR-10 mirror does not match official length/MD5")
        os.replace(partial, TARGET)
    if TARGET.stat().st_size != EXPECTED_LENGTH or md5(TARGET) != EXPECTED_MD5:
        raise RuntimeError(
            "existing CIFAR-10 archive does not match official length/MD5"
        )
    result = {
        "mirror": URL,
        "official_checksum_page": OFFICIAL,
        "local": str(TARGET),
        "bytes": EXPECTED_LENGTH,
        "md5": EXPECTED_MD5,
        "reused_verified_file": reused,
        "scope": "official-MD5 Python archive, no members extracted",
    }
    AUDIT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
