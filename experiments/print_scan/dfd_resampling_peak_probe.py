"""Check whether publication-like resizing can erase genuine print lattices.

Uses already verified 800-PPI DFD gray scans, identical 512-PPI physical
regions and fixed 1x/2x/4x downsampling. This is not a reconstruction of the
unpublished DESCAN preprocessing, only a controlled processing counterexample.
"""

from __future__ import annotations

from experiments.paths import WORK_DIR

import json
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE = Path(r"E:\ai_image_origin_research\data\derived\dfd_probe")
MEASUREMENTS = WORK_DIR / "dfd_tone_lattice.json"
OUTPUT = WORK_DIR / "dfd_resampling_peak_probe.json"


def peak_ratio(image: np.ndarray) -> dict:
    h, w = image.shape
    centered = image.astype(np.float64) - np.mean(image)
    power = (
        np.abs(
            np.fft.fftshift(
                np.fft.fft2(centered * np.outer(np.hanning(h), np.hanning(w)))
            )
        )
        ** 2
    )
    yy, xx = np.ogrid[-h // 2 : h // 2, -w // 2 : w // 2]
    radius = np.sqrt((xx / w) ** 2 + (yy / h) ** 2)
    band = (radius >= 0.08) & (radius <= 0.42)
    masked = np.where(band, power, 0)
    py, px = np.unravel_index(np.argmax(masked), power.shape)
    peak = masked[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2].sum()
    return {
        "peak_share": float(peak / (masked.sum() + 1e-12)),
        "peak_to_band_median": float(power[py, px] / (np.median(power[band]) + 1e-12)),
        "fx_fy": [(px - w // 2) / w, (py - h // 2) / h],
        "std": float(np.std(image) / 255),
    }


def main() -> None:
    audit = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    rows = []
    for device in ("D5", "D6"):
        path = SOURCE / f"{device}_DC1_x_800_P3_S1_T1_2111_1.tiff"
        with Image.open(path) as full:
            for name, item in audit["measurements"][device].items():
                if not 130 <= item["radial_lpi"] <= 150:
                    continue
                patch = full.crop(tuple(item["roi_xyxy"]))
                record = {
                    "device": device,
                    "tile": name,
                    "original_lpi_peak": item["radial_lpi"],
                }
                for size in (512, 256, 128):
                    resized = (
                        patch
                        if size == 512
                        else patch.resize((size, size), Image.Resampling.LANCZOS)
                    )
                    array = np.asarray(resized.convert("L"))
                    record[str(size)] = peak_ratio(array)
                rows.append(record)
    summary = {
        str(size): {
            key: float(np.median([r[str(size)][key] for r in rows]))
            for key in ("peak_share", "peak_to_band_median", "std")
        }
        for size in (512, 256, 128)
    }
    result = {
        "rows": rows,
        "summary": summary,
        "method": "D5/D6 verified 800ppi gray scans; eight lighter tone tiles per device; Lanczos same-FOV resize",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
