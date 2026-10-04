"""Read TAR member headers only via HTTP Range; never unpack whole archive."""

from palimpsest.paths import WORK_DIR

import json
import tarfile

import requests

from experiments.data_preparation.raw2event.prepare_download_first_raw_tar_member import (
    URL,
)


def main() -> None:
    offset = 0
    entries = []
    for _ in range(5):
        response = requests.get(
            URL, headers={"Range": f"bytes={offset}-{offset + 4095}"}, timeout=90
        )
        if response.status_code != 206 or len(response.content) != 4096:
            raise RuntimeError("short TAR header Range")
        first = tarfile.TarInfo.frombuf(
            response.content[:512], encoding="utf-8", errors="replace"
        )
        if first.type == tarfile.XHDTYPE:
            file_offset = offset + 512 + ((first.size + 511) // 512) * 512
            local_offset = file_offset - offset
            actual = tarfile.TarInfo.frombuf(
                response.content[local_offset : local_offset + 512],
                encoding="utf-8",
                errors="replace",
            )
        else:
            file_offset = offset
            actual = first
        entries.append(
            {"header_offset": file_offset, "name": actual.name, "size": actual.size}
        )
        offset = file_offset + 512 + ((actual.size + 511) // 512) * 512
    out = WORK_DIR / "raw2event_raw_tar_first_headers.json"
    out.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
