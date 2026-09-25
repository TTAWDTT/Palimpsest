"""Read the beginning of the official uncompressed metadata TAR by HTTP Range."""

import io
import hashlib
import json
from pathlib import Path
import tarfile

import requests


URL = ("https://huggingface.co/datasets/raw2event/raw2event/resolve/"
       "9df99d9ed09e5ed705f50cae011e49cb2af620b9/metadata/metadata-0001-of-0001.tar")
ROOT = Path("E:/ai_image_origin_research/data/raw/raw2event_probe")
OUT = Path("work/raw2event_metadata_tar_range_audit.json")


def main() -> None:
    response = requests.get(URL, headers={"Range": "bytes=0-4194303"}, timeout=60)
    if response.status_code != 206 or len(response.content) != 4_194_304:
        raise RuntimeError("metadata TAR range not returned exactly")
    entries = []
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:") as archive:
        for member in archive:
            if member.offset_data + member.size > len(response.content):
                break
            item = {"name": member.name, "size": member.size, "type": member.type.decode("ascii", errors="replace")}
            if member.isfile() and member.size <= 100_000:
                with archive.extractfile(member) as stream:
                    payload = stream.read()
                item["sha256"] = hashlib.sha256(payload).hexdigest()
                item["record_stride_38"] = member.size % 38 == 0
                local = ROOT / "meta_raw" / Path(member.name).name.removeprefix("metadata_")
                item["local_sidecar_exists"] = local.exists()
                if local.exists():
                    item["local_sidecar_sha256"] = hashlib.sha256(local.read_bytes()).hexdigest()
                    item["identical_to_meta_raw"] = item["local_sidecar_sha256"] == item["sha256"]
            entries.append(item)
            if len(entries) >= 30:
                break
    report = {"tar_total_bytes_reported_by_hf": 821_893_120,
              "range_bytes": len(response.content), "first_members": entries,
              "scope": "only first 4 MiB TAR range; no full archive audit"}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
