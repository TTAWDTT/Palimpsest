"""Range-extract only the first raw TAR member, not its 17.9 GB archive."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import requests


URL = (
    "https://huggingface.co/datasets/raw2event/raw2event/resolve/"
    "9df99d9ed09e5ed705f50cae011e49cb2af620b9/raw/raw-0001-of-0250.tar"
)
ROOT = Path("E:/ai_image_origin_research/data/raw/raw2event_probe/raw_tar_member")
OUT = Path("work/raw2event_first_raw_tar_member_audit.json")


def main() -> None:
    header = requests.get(URL, headers={"Range": "bytes=0-4095"}, timeout=90)
    if header.status_code != 206 or len(header.content) != 4096:
        raise RuntimeError("remote TAR header Range did not return 4096 bytes")
    pax = tarfile.TarInfo.frombuf(
        header.content[:512], encoding="utf-8", errors="replace"
    )
    member = tarfile.TarInfo.frombuf(
        header.content[1024:1536], encoding="utf-8", errors="replace"
    )
    if pax.type != tarfile.XHDTYPE or member.name != (
        "raw/raw_frames_10000_automobile_5_1087_20251224_105416_raw_10bit.mkv"
    ):
        raise RuntimeError("unexpected first TAR member")
    start, end = 1536, 1536 + member.size - 1
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / Path(member.name).name
    if target.exists():
        if target.stat().st_size != member.size:
            raise RuntimeError("existing target has wrong length")
    else:
        partial = target.with_suffix(".mkv.partial")
        with requests.get(
            URL,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=(20, 120),
        ) as response:
            if response.status_code != 206 or response.headers.get("Content-Range") != (
                f"bytes {start}-{end}/{header.headers['Content-Range'].split('/')[-1]}"
            ):
                raise RuntimeError("payload Range has incorrect content bounds")
            digest = hashlib.sha256()
            count = 0
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        count += len(chunk)
                        if count > member.size:
                            raise RuntimeError("remote payload exceeds member size")
                        stream.write(chunk)
                        digest.update(chunk)
        if count != member.size:
            raise RuntimeError(f"short payload: {count} vs {member.size}")
        partial.replace(target)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(target)],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    report = {
        "repository_revision": "9df99d9ed09e5ed705f50cae011e49cb2af620b9",
        "archive_url": URL,
        "archive_total_bytes": int(header.headers["Content-Range"].split("/")[-1]),
        "member": member.name,
        "member_payload_start_end": [start, end],
        "member_size": member.size,
        "local_file": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "stream": {
            key: stream.get(key)
            for key in (
                "codec_name",
                "width",
                "height",
                "pix_fmt",
                "r_frame_rate",
                "avg_frame_rate",
                "bits_per_raw_sample",
            )
        },
        "qualification": "TAR header and exact HTTP Range length checked; whole 17.9 GB archive and LFS SHA not downloaded",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
