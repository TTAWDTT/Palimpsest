"""Audit DFD's public B/W scanned-sheet archive without unpacking it.

The fixed ROI is within one light gray patch on the P3 chart for 15 D5/D6
scans of the same named printer model. It is an observational probe, not a full
input-to-output print model. The archive must first have the HTTP HEAD size.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ARCHIVE = Path(
    r"E:\ai_image_origin_research\data\raw\dfd_halftone\HalftoneImages-BW.tar.gz"
)
OUT = Path("work/dfd_halftone_audit.json")
EXPECTED_BYTES = 4_259_915_309
ROI = (5100, 1800, 5612, 2312)
SHARED_SUFFIXES = (
    "DC1_x_800_P3_S1_T1_2111_1.tiff",
    "DC2_Normal_800_P3_S1_T1_2111_1.tiff",
    "W1_R36_800_P3_S1_T1_2111_3.tiff",
    "DC2_Best_800_P3_S1_T1_2111_3.tiff",
    "W1_600_800_P3_S1_T1_2111_1.tiff",
    "W1_FR12_800_P3_S1_T1_2111_2.tiff",
    "DC2_Draft_800_P3_S1_T1_2111_2.tiff",
)
PROBE_NAMES = (
    {f"D5_{suffix}" for suffix in SHARED_SUFFIXES}
    | {f"D6_{suffix}" for suffix in SHARED_SUFFIXES}
    | {"D6_W1_FPA_800_P3_S1_T1_2111_4.tiff"}
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spectrum_probe(data: bytes, name: str) -> dict:
    with Image.open(io.BytesIO(data)) as image:
        shape = list(image.size)
        mode = image.mode
        dpi = image.info.get("dpi")
        dpi = [float(value) for value in dpi] if dpi is not None else None
        patch = np.asarray(image.crop(ROI), dtype=np.float64)
    if patch.shape != (512, 512):
        raise ValueError(f"probe ROI outside image: {name}")
    window = np.hanning(512)
    demeaned = (patch - patch.mean()) * window[:, None] * window[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(demeaned)))
    yy, xx = np.indices((512, 512))
    spectrum[np.hypot(xx - 256, yy - 256) < 20] = 0
    row, col = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    fx, fy = (col - 256) / 512, (row - 256) / 512
    return {
        "name": name,
        "width_height": shape,
        "mode": mode,
        "dpi": dpi,
        "roi_xyxy": ROI,
        "roi_mean": float(patch.mean()),
        "roi_std": float(patch.std()),
        "strongest_peak_cycles_per_scan_pixel_xy": [fx, fy],
        "strongest_peak_radial_lpi_at_recorded_dpi": float(np.hypot(fx, fy) * dpi[0])
        if dpi
        else None,
    }


def main() -> None:
    actual = ARCHIVE.stat().st_size
    if actual != EXPECTED_BYTES:
        raise RuntimeError(f"archive incomplete: {actual} of {EXPECTED_BYTES} bytes")
    report = {
        "archive": str(ARCHIVE),
        "archive_bytes": actual,
        "archive_sha256": sha256(ARCHIVE),
        "expected_bytes_source": "official download HTTP HEAD on 2026-09-24",
        "archive_integrity": "pending full gzip/tar stream traversal",
        "member_count": 0,
        "tiff_count": 0,
        "folder_counts": {},
        "tiff_members": [],
        "non_tiff_files": [],
        "probes": [],
        "probe_scope": "15 D5/D6 P3 sheets, fixed gray patch ROI; strongest observed FFT peak is not necessarily the RIP's fundamental screen frequency",
    }
    folder_counts = Counter()
    with tarfile.open(ARCHIVE, mode="r|gz") as archive:
        for member in archive:
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            report["member_count"] += 1
            if not member.isfile():
                continue
            if member.name.lower().endswith((".tif", ".tiff")):
                report["tiff_count"] += 1
                report["tiff_members"].append(
                    {"name": member.name, "bytes": member.size}
                )
                parts = Path(member.name).parts
                if len(parts) >= 3:
                    folder_counts[parts[1]] += 1
            else:
                report["non_tiff_files"].append(
                    {"name": member.name, "bytes": member.size}
                )
            if Path(member.name).name in PROBE_NAMES:
                with archive.extractfile(member) as source:
                    data = source.read()
                if len(data) != member.size:
                    raise ValueError(f"truncated TIFF: {member.name}")
                report["probes"].append(spectrum_probe(data, member.name))
    decompressed_bytes = 0
    with gzip.open(ARCHIVE, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            decompressed_bytes += len(chunk)
    report["folder_counts"] = dict(sorted(folder_counts.items()))
    report["decompressed_bytes"] = decompressed_bytes
    report["archive_integrity"] = "gzip trailer/CRC checked and tar traversal completed"
    if len(report["probes"]) != len(PROBE_NAMES):
        raise ValueError(
            f"found {len(report['probes'])} of {len(PROBE_NAMES)} expected D5 probes"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "archive_bytes",
                    "archive_sha256",
                    "member_count",
                    "tiff_count",
                    "folder_counts",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
