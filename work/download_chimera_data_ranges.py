"""Download the public Zenodo data archive with checked independent byte ranges.

The fixed official size and MD5 are taken from the Zenodo record. Range parts
stay on E: until the assembled archive passes its whole-file digest.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request


URL = "https://zenodo.org/api/records/14736478/files/data.tar.gz/content"
EXPECTED_SIZE = 3_784_718_041
EXPECTED_MD5 = "e645697149fd75afe6382131020cc394"
BASE = Path("E:/ai_image_origin_research/data/raw")
PARTIAL = BASE / "chimera_data.tar.gz.partial"
FINAL = BASE / "chimera_data.tar.gz"
PARTS = BASE / "chimera_data_range_parts"
CHUNK = 64 * 1024 * 1024


def get_part(index: int, start: int, end: int) -> tuple[int, int]:
    target = PARTS / f"part_{index:04d}.bin"
    expected_length = end - start + 1
    if target.exists() and target.stat().st_size == expected_length:
        return index, expected_length
    temporary = PARTS / f"part_{index:04d}.partial"
    for attempt in range(5):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"range {index}: status {response.status}")
                actual_range = response.headers.get("Content-Range", "")
                wanted_range = f"bytes {start}-{end}/{EXPECTED_SIZE}"
                if actual_range != wanted_range:
                    raise RuntimeError(f"range {index}: {actual_range!r} != {wanted_range!r}")
                written = 0
                with temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        written += len(block)
                if written != expected_length:
                    raise RuntimeError(f"range {index}: got {written}, expected {expected_length}")
            temporary.replace(target)
            return index, written
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    if FINAL.exists():
        if FINAL.stat().st_size == EXPECTED_SIZE:
            print("final archive already present; verify its MD5 separately", flush=True)
            return
        raise RuntimeError("final archive exists with unexpected size")
    prefix_length = PARTIAL.stat().st_size if PARTIAL.exists() else 0
    if prefix_length >= EXPECTED_SIZE:
        raise RuntimeError("partial archive already reaches expected size")
    jobs = []
    for index, start in enumerate(range(prefix_length, EXPECTED_SIZE, CHUNK)):
        jobs.append((index, start, min(start + CHUNK, EXPECTED_SIZE) - 1))
    print(f"prefix={prefix_length} bytes; {len(jobs)} range jobs", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(get_part, *job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            index, written = future.result()
            print(f"range {index + 1}/{len(jobs)} ready ({written} bytes); "
                  f"completed {completed}/{len(jobs)}", flush=True)
    with PARTIAL.open("ab") as output:
        for index, start, end in jobs:
            target = PARTS / f"part_{index:04d}.bin"
            if target.stat().st_size != end - start + 1:
                raise RuntimeError(f"range {index} size changed before assembly")
            with target.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        output.flush()
    if PARTIAL.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(f"assembled size {PARTIAL.stat().st_size} != {EXPECTED_SIZE}")
    digest = hashlib.md5()
    with PARTIAL.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    actual_md5 = digest.hexdigest()
    print(f"whole-file MD5={actual_md5}", flush=True)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError("whole-file MD5 mismatch; preserve parts for audit")
    PARTIAL.replace(FINAL)
    print(f"VERIFIED {FINAL} ({EXPECTED_SIZE} bytes)", flush=True)


if __name__ == "__main__":
    main()
