"""Range-extract one CSGC source/scan pair per SPI and compare templates."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


DEST = Path(r"E:\ai_image_origin_research\data\derived\csgc_probe")
DEST.mkdir(parents=True, exist_ok=True)


def extract(spi: str, folder: str, number: int = 1) -> tuple[Path, dict]:
    index = json.loads(Path(f"work/csgc{spi}_remote_index.json").read_text(encoding="utf-8"))
    name = f"{spi}dpi_NEW/{folder}/Scan{number:05d}.tif"
    entry = next(r for r in index["records"] if r["name"] == name)
    target = DEST / name.replace("/", "__")
    if target.exists():
        raw = target.read_bytes()
    else:
        start = entry["local_header_offset"]
        end = start + 1024 + entry["compressed_bytes"] - 1
        request = urllib.request.Request(index["url"], headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError(f"{name}: expected 206, got {response.status}")
            packet = response.read()
        if len(packet) != end - start + 1:
            raise RuntimeError(f"{name}: short range response")
        header = struct.unpack_from("<IHHHHHIIIHH", packet)
        if header[0] != 0x04034B50 or entry["method"] != 8:
            raise RuntimeError(f"{name}: unsupported local header or method")
        offset = 30 + header[-2] + header[-1]
        raw = zlib.decompress(packet[offset:offset + entry["compressed_bytes"]], -15)
        target.write_bytes(raw)
    if len(raw) != entry["uncompressed_bytes"] or zlib.crc32(raw) != int(entry["crc32"], 16):
        raise RuntimeError(f"{name}: entry length/CRC mismatch")
    with Image.open(io.BytesIO(raw)) as image:
        array = np.asarray(image)
        encoded_dpi = image.info.get("dpi")
        summary = {"name": name, "sha256": hashlib.sha256(raw).hexdigest(),
                   "shape": list(array.shape), "mode": image.mode,
                   "encoded_dpi": [float(x) for x in encoded_dpi] if encoded_dpi else None,
                   "min": int(array.min()), "max": int(array.max())}
    return target, summary


def main() -> None:
    summary = {}
    templates = {}
    for spi, factor in (("2400", 4), ("4800", 8), ("9600", 16)):
        binary_path, binary_record = extract(spi, f"bin_exact_crop_{factor}")
        _, scan_record = extract(spi, "resize_exact_crop")
        with Image.open(binary_path) as image:
            pattern = np.asarray(image)
        if pattern.shape != (100 * factor, 100 * factor):
            raise RuntimeError(f"{spi}: binary crop not 100x100 upscaled by {factor}")
        blocks = pattern.reshape(100, factor, 100, factor)
        block_uniform_fraction = float(np.mean(blocks.max(axis=(1, 3)) == blocks.min(axis=(1, 3))))
        templates[spi] = pattern[::factor, ::factor]
        summary[spi] = {"binary": binary_record, "scan": scan_record,
                        "binary_block_uniform_fraction": block_uniform_fraction}
    summary["template_sha256_100x100"] = {
        spi: hashlib.sha256(template.tobytes()).hexdigest() for spi, template in templates.items()}
    summary["all_templates_equal"] = bool(
        np.array_equal(templates["2400"], templates["4800"]) and
        np.array_equal(templates["2400"], templates["9600"]))
    Path("work/csgc_cross_resolution_probe.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
