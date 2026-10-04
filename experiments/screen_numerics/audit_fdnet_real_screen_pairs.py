"""Fetch and verify the small real-screen focus/defocus examples in FDNet authors' repo."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import requests
from PIL import Image


REPO = "baolp/demoireing_with_focused_and_defocused_images_pairs"
ROOT = Path("E:/ai_image_origin_research/data/raw/fdnet_real_screen_pairs")
OUT = Path("work/fdnet_real_screen_pairs_audit.json")
PREFIX = "datasets/realscreenmoire/"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> None:
    session = requests.Session()
    head = session.get(f"https://api.github.com/repos/{REPO}/commits/main", timeout=40)
    head.raise_for_status()
    commit = head.json()["sha"]
    tree_response = session.get(
        f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1",
        timeout=60,
    )
    tree_response.raise_for_status()
    tree = tree_response.json()
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree truncated")
    entries = [
        item
        for item in tree["tree"]
        if item["type"] == "blob" and item["path"].startswith(PREFIX)
    ]
    groups = {"moireresize": {}, "blurresize": {}}
    records = []
    for item in entries:
        relative = item["path"][len(PREFIX) :]
        condition, name = relative.split("/", 1)
        if condition not in groups or not name.lower().endswith(".png"):
            raise RuntimeError(f"unexpected real-screen entry {relative}")
        groups[condition][name] = item
    if not groups["moireresize"] or set(groups["moireresize"]) != set(
        groups["blurresize"]
    ):
        raise RuntimeError("real screen pair name sets differ")
    for condition in groups:
        for name in sorted(groups[condition]):
            item = groups[condition][name]
            url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{item['path']}"
            response = session.get(url, timeout=90)
            response.raise_for_status()
            data = response.content
            if len(data) != item["size"] or git_blob_sha(data) != item["sha"]:
                raise RuntimeError(f"Git blob verification failed: {item['path']}")
            folder = ROOT / condition
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / name
            path.write_bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                dimensions = list(image.size)
                mode = image.mode
                info_keys = sorted(image.info)
            records.append(
                {
                    "path": item["path"],
                    "condition": condition,
                    "name": name,
                    "git_blob_sha1": item["sha"],
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "dimensions_wh": dimensions,
                    "mode": mode,
                    "info_keys": info_keys,
                    "local_path": str(path),
                }
            )
    report = {
        "repository": f"https://github.com/{REPO}",
        "commit": commit,
        "counts": {key: len(value) for key, value in groups.items()},
        "n_pairs_by_matching_filename": len(groups["moireresize"]),
        "pairing_status": "filename only; content registration to be audited separately",
        "records": records,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "repository",
                    "commit",
                    "counts",
                    "n_pairs_by_matching_filename",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
