"""Fixed-parameter color-print prototype versus selected real DESCAN tiles.

This is a falsification probe, not calibration: DESCAN lacks physical print
scale, RIP, paper and scanner settings; 300/1200/600 PPI are nominal choices.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from origin_simulation.color_print_scan import ColorPrintScanParameters, simulate_color_print_scan
from origin_simulation.publication import PublicationParameters, apply_publication


BASE = Path(__file__).resolve().parent
DATA = Path(r"E:\ai_image_origin_research\data\raw\descan18k")


def _read(archive: zipfile.ZipFile, path: str) -> np.ndarray:
    with Image.open(io.BytesIO(archive.read(path))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _peak_concentration(rgb: np.ndarray) -> float:
    image = np.asarray(rgb, dtype=np.float32) / 255
    luma = .2126 * image[:, :, 0] + .7152 * image[:, :, 1] + .0722 * image[:, :, 2]
    window = np.outer(np.hanning(luma.shape[0]), np.hanning(luma.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2((luma-luma.mean())*window)))**2
    yy, xx = np.ogrid[-64:64, -64:64]
    radius = np.sqrt(xx**2 + yy**2) / 128
    band = (radius >= .08) & (radius <= .42)
    # Fraction of annular power in the strongest 3x3 peak neighborhood.
    masked = np.where(band, power, 0)
    py, px = np.unravel_index(np.argmax(masked), masked.shape)
    neighborhood = masked[max(0,py-1):py+2, max(0,px-1):px+2]
    return float(neighborhood.sum() / (masked.sum() + 1e-12))


def main() -> None:
    selection = json.loads((BASE / "descan_print_signatures.json").read_text(encoding="utf-8"))["rows"]
    by_scanner: dict[str, list[dict]] = defaultdict(list)
    for row in selection:
        by_scanner[row["scanner"]].append(row)
    chosen = []
    for scanner, rows in sorted(by_scanner.items()):
        # Deterministic, one tile per page so one document cannot dominate.
        seen = set()
        for row in rows:
            if row["file"] in seen:
                continue
            seen.add(row["file"])
            chosen.append(row)
            if len(seen) == 4:
                break
    parameter = replace(ColorPrintScanParameters(), paper_scatter_sigma_um=10.,
                        scanner_optical_sigma_um=8.)
    output = []
    with zipfile.ZipFile(DATA / "Valid.zip") as valid, zipfile.ZipFile(DATA / "Test.zip") as test:
        archives = {"Valid": valid, "Test": test}
        for row in chosen:
            archive = archives[row["split"]]
            stem = f"{row['split']}/"
            clean = _read(archive, stem + "clean/" + row["file"])
            scan = _read(archive, stem + "scan/" + row["file"])
            top, left = row["y"], row["x"]
            source_tile = clean[top:top+128, left:left+128]
            real_tile = scan[top:top+128, left:left+128]
            simulated = simulate_color_print_scan(source_tile, parameter)
            published = apply_publication(simulated.scanner_output_rgb,
                                          PublicationParameters(output_size=(128, 128),
                                                                resampling="bicubic", encoding="png"))
            sim_tile = published.decoded_rgb
            output.append({"scanner": row["scanner"], "file": row["file"], "x": left, "y": top,
                           "source_peak_share": _peak_concentration(source_tile),
                           "real_peak_share": _peak_concentration(real_tile),
                           "sim_peak_share": _peak_concentration(sim_tile),
                           "real_rgb_mean": (real_tile / 255).mean((0,1)).tolist(),
                           "sim_rgb_mean": (sim_tile / 255).mean((0,1)).tolist()})
    result = {"parameters": parameter.__dict__, "selection": "first tile on four distinct pages per scanner from frozen content-gated list",
              "rows": output,
              "median_peak_share": {group: float(np.median([row[group] for row in output]))
                                    for group in ("source_peak_share", "real_peak_share", "sim_peak_share")}}
    (BASE / "descan_color_forward_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["median_peak_share"], indent=2))


if __name__ == "__main__":
    main()
