"""Download and verify selected official Raw2Event RAW/RGB/metadata prefixes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import requests


REPO = "raw2event/raw2event"
DEFAULT_PREFIX = "10000_automobile_5_1087_20251224_105416"
ROOT = Path("E:/ai_image_origin_research/data/raw/raw2event_probe")
DIRS = ("frames_raw", "frames_rgb", "meta_raw")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(b"blob " + str(path.stat().st_size).encode("ascii") + bytes([0]))
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def remote_metadata(session: requests.Session, prefix: str, revision: str = "main") -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{REPO}/paths-info/{revision}"
    expected = [f"{directory}/{prefix}.{'dat' if directory == 'meta_raw' else 'mkv'}"
                for directory in DIRS]
    response = session.post(url, json={"paths": expected}, timeout=30)
    response.raise_for_status()
    items = {item["path"]: item for item in response.json()}
    if set(items) != set(expected):
        raise RuntimeError(f"official prefix has missing or extra paths: {expected}, {list(items)}")
    return [items[path] for path in expected]


def fetch_one(session: requests.Session, metadata: dict, revision: str = "main") -> dict:
    path = ROOT / metadata["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(metadata["size"])
    expected_sha = metadata.get("lfs", {}).get("oid")
    expected_blob = metadata.get("oid") if not expected_sha else None
    if path.exists():
        digest = sha256(path)
        if (path.stat().st_size != expected_size or (expected_sha and digest != expected_sha)
                or (expected_blob and git_blob_sha1(path) != expected_blob)):
            raise RuntimeError(f"existing file is not the official object: {path}")
        return {"path": str(path), "size": expected_size, "sha256": digest,
                "official_lfs_sha256": expected_sha, "official_git_blob_sha1": expected_blob,
                "reused_verified_file": True}
    partial = path.with_suffix(path.suffix + ".partial")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{metadata['path']}"
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise RuntimeError(f"partial file exceeds expected size: {partial}")
        if offset == expected_size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(url, stream=True, headers=headers, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError(f"server did not honor resume range for {metadata['path']}")
                if not offset and response.status_code != 200:
                    raise RuntimeError(f"unexpected HTTP status for {metadata['path']}")
                mode = "ab" if offset else "wb"
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(10, 1 + 2 ** attempt))
            continue
        if partial.stat().st_size == expected_size:
            break
        if attempt == 5:
            raise RuntimeError(f"incomplete after retries: {metadata['path']}")
        time.sleep(min(10, 1 + 2 ** attempt))
    if not partial.exists() or partial.stat().st_size != expected_size:
        raise RuntimeError(f"incomplete official file: {metadata['path']}")
    digest = sha256(partial)
    if expected_sha and digest != expected_sha:
        raise RuntimeError(f"SHA256 mismatch: {metadata['path']}")
    if expected_blob and git_blob_sha1(partial) != expected_blob:
        raise RuntimeError(f"Git blob SHA1 mismatch: {metadata['path']}")
    os.replace(partial, path)
    return {"path": str(path), "size": expected_size, "sha256": digest,
            "official_lfs_sha256": expected_sha, "official_git_blob_sha1": expected_blob,
            "reused_verified_file": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--audit", type=Path, default=Path("work/raw2event_probe_download_audit.json"))
    args = parser.parse_args()
    session = requests.Session()
    audit = {"repo": REPO, "revision": args.revision, "prefix": args.prefix,
             "scope": "one paired RAW/RGB/meta recording",
             "source": f"https://huggingface.co/datasets/{REPO}", "files": []}
    for metadata in remote_metadata(session, args.prefix, args.revision):
        record = fetch_one(session, metadata, args.revision)
        audit["files"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
