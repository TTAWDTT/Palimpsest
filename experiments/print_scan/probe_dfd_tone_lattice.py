"""Observed peak stability across gray tiles in two matched DFD P3 scans."""

from __future__ import annotations

import json
import hashlib
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


ARCHIVE = Path(
    r"E:\ai_image_origin_research\data\raw\dfd_halftone\HalftoneImages-BW.tar.gz"
)
DERIVED = Path(r"E:\ai_image_origin_research\data\derived\dfd_probe")
D6_NAME = "HalftoneImages-BW/D6_HPCLJ5550/D6_DC1_x_800_P3_S1_T1_2111_1.tiff"
TILE_X = [1000, 2000, 3000, 4000, 5000]
TILE_Y = [700, 1800]


def find_peak(patch: np.ndarray) -> dict:
    p = np.asarray(patch, dtype=np.float64)
    window = np.hanning(512)
    spectrum = np.abs(
        np.fft.fftshift(np.fft.fft2((p - p.mean()) * window[:, None] * window[None, :]))
    )
    yy, xx = np.indices((512, 512))
    radius = np.hypot(xx - 256, yy - 256)
    annulus = (radius > 60) & (radius < 150)
    spectrum[radius < 20] = 0
    row, col = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    return {
        "mean": float(p.mean()),
        "std": float(p.std()),
        "fx_fy_cycles_per_px": [(col - 256) / 512, (row - 256) / 512],
        "radial_lpi": float(np.hypot(col - 256, row - 256) / 512 * 800),
        "peak_to_annulus_median": float(
            spectrum[row, col] / np.median(spectrum[annulus])
        ),
    }


def main() -> None:
    d6 = DERIVED / Path(D6_NAME).name
    if not d6.exists():
        with tarfile.open(ARCHIVE, mode="r|gz") as tar:
            for member in tar:
                if member.name == D6_NAME:
                    with tar.extractfile(member) as source, d6.open("wb") as sink:
                        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            sink.write(chunk)
                    if d6.stat().st_size != member.size:
                        raise ValueError("D6 extraction truncated")
                    break
            else:
                raise FileNotFoundError(D6_NAME)
    result = {"source_sha256": {}, "measurements": {}}
    for device in ("D5", "D6"):
        path = DERIVED / f"{device}_DC1_x_800_P3_S1_T1_2111_1.tiff"
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result["source_sha256"][device] = digest.hexdigest()
        with Image.open(path) as image:
            patches = {}
            for row, y in enumerate(TILE_Y):
                for col, x in enumerate(TILE_X):
                    patches[f"r{row + 1}c{col + 1}"] = {
                        "roi_xyxy": [x, y, x + 512, y + 512],
                        **find_peak(image.crop((x, y, x + 512, y + 512))),
                    }
            result["measurements"][device] = patches
    Path("work/dfd_tone_lattice.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for device, patches in result["measurements"].items():
        print(device)
        for tile, values in patches.items():
            print(
                tile,
                round(values["mean"], 1),
                round(values["radial_lpi"], 2),
                round(values["peak_to_annulus_median"], 1),
            )


if __name__ == "__main__":
    main()
