"""Range-extract one independently selected RAW TAR member with a known direct counterpart."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import requests

from experiments.raw2event.fetch_raw2event_first_raw_tar_member import URL, ROOT


TARGET_PREFIX = "10034_automobile_5_1356_20251224_110057"
OUT = Path("work/raw2event_second_raw_tar_member_audit.json")


def main() -> None:
    session = requests.Session()
    offset = 0
    headers_read = 0
    while headers_read < 80:
        response = session.get(
            URL, headers={"Range": f"bytes={offset}-{offset + 2047}"}, timeout=90
        )
        if response.status_code != 206 or len(response.content) != 2048:
            raise RuntimeError("remote TAR header Range failed")
        pax = tarfile.TarInfo.frombuf(
            response.content[:512], encoding="utf-8", errors="replace"
        )
        if pax.type != tarfile.XHDTYPE:
            raise RuntimeError(f"expected PAX header at {offset}")
        header_offset = offset + 512 + ((pax.size + 511) // 512) * 512
        local_offset = header_offset - offset
        member = tarfile.TarInfo.frombuf(
            response.content[local_offset : local_offset + 512],
            encoding="utf-8",
            errors="replace",
        )
        headers_read += 1
        if member.name == f"raw/raw_frames_{TARGET_PREFIX}_raw_10bit.mkv":
            break
        offset = header_offset + 512 + ((member.size + 511) // 512) * 512
    else:
        raise RuntimeError("target member not found in first 80 TAR records")
    start = header_offset + 512
    end = start + member.size - 1
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / Path(member.name).name
    if target.exists() and target.stat().st_size != member.size:
        raise RuntimeError("existing target length mismatch")
    if not target.exists():
        partial = target.with_suffix(".mkv.partial")
        with session.get(
            URL,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=(20, 120),
        ) as payload:
            total = response.headers["Content-Range"].split("/")[-1]
            if (
                payload.status_code != 206
                or payload.headers.get("Content-Range")
                != f"bytes {start}-{end}/{total}"
            ):
                raise RuntimeError("payload Content-Range mismatch")
            count = 0
            with partial.open("wb") as stream:
                for chunk in payload.iter_content(chunk_size=1 << 20):
                    if chunk:
                        count += len(chunk)
                        if count > member.size:
                            raise RuntimeError("payload too large")
                        stream.write(chunk)
            if count != member.size:
                raise RuntimeError("payload too short")
        partial.replace(target)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(target)],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    report = {
        "target_prefix": TARGET_PREFIX,
        "archive_url": URL,
        "headers_read": headers_read,
        "tar_header_offset": header_offset,
        "member": member.name,
        "payload_start_end": [start, end],
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
            )
        },
        "qualification": "exact TAR header/HTTP Range checked, archive-wide LFS SHA not verified",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
