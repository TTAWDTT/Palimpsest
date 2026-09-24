"""Index all three CSGC official ZIPs from verified HTTP Range tail fragments."""

import json
import struct
from collections import Counter
from pathlib import Path


SIZES = {"2400": 113_349_148, "4800": 370_363_420, "9600": 1_208_354_427}


def index_one(spi: str, total_bytes: int) -> dict:
    tail = Path(f"work/csgc{spi}_tail.bin").read_bytes()
    tail_offset = total_bytes - len(tail)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError(f"{spi}: no EOCD in tail")
    _, disk, cd_disk, disk_entries, n_entries, cd_size, cd_offset, _ = struct.unpack_from(
        "<IHHHHIIH", tail, eocd)
    if disk or cd_disk or disk_entries != n_entries or not (
        tail_offset <= cd_offset and cd_offset + cd_size <= total_bytes
    ):
        raise RuntimeError(f"{spi}: unsupported ZIP layout")
    cursor = cd_offset - tail_offset
    records = []
    for _ in range(n_entries):
        fields = struct.unpack_from("<IHHHHHHIIIHHHHHII", tail, cursor)
        if fields[0] != 0x02014B50:
            raise RuntimeError(f"{spi}: invalid central entry")
        (_, _, _, flags, method, _, _, crc32, compressed, uncompressed,
         name_len, extra_len, comment_len, _, _, _, local_offset) = fields
        start = cursor + 46
        name = tail[start:start + name_len].decode("utf-8" if flags & 0x800 else "cp437")
        records.append({"name": name, "method": method, "crc32": f"{crc32:08x}",
                        "compressed_bytes": compressed, "uncompressed_bytes": uncompressed,
                        "local_header_offset": local_offset})
        cursor = start + name_len + extra_len + comment_len
    if cursor != cd_offset - tail_offset + cd_size:
        raise RuntimeError(f"{spi}: central directory byte count mismatch")
    report = {
        "url": f"https://www.univ-st-etienne.fr/graphical-code-estimation/CSGC_scan{spi}spi.zip",
        "archive_bytes_http_head": total_bytes,
        "entry_count": n_entries,
        "central_directory_offset": cd_offset,
        "central_directory_bytes": cd_size,
        "extension_counts": dict(Counter(Path(r["name"]).suffix.lower() for r in records)),
        "folder_counts": dict(Counter("/".join(r["name"].split("/")[:-1]) for r in records
                                      if r["name"].endswith(".tif"))),
        "records": records,
    }
    Path(f"work/csgc{spi}_remote_index.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {k: report[k] for k in ("entry_count", "extension_counts", "folder_counts")}


if __name__ == "__main__":
    print(json.dumps({spi: index_one(spi, size) for spi, size in SIZES.items()}, indent=2))
