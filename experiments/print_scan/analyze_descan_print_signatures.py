"""Predefined, content-gated DESCAN print/scan observations; no model fit.

Reads the official Valid/Test ZIPs in place. Selects smooth clean 128-pixel
tiles before looking at their scanned counterparts; reports distributions by
scanner, without inferring unknown physical scanner/print DPI.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = DATA_ROOT / "raw/descan18k"
OUTPUT = WORK_DIR / "descan_print_signatures.json"
TILE = 128


def _image(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with Image.open(io.BytesIO(archive.read(name))) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def _high_band(image: np.ndarray) -> float:
    centered = image - image.mean()
    window = np.outer(np.hanning(TILE), np.hanning(TILE))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    yy, xx = np.ogrid[-TILE // 2 : TILE // 2, -TILE // 2 : TILE // 2]
    radius = np.sqrt(xx * xx + yy * yy) / TILE
    band = (radius >= 0.18) & (radius <= 0.42)
    return float(
        np.mean(spectrum[band])
        / (np.mean(spectrum[(radius >= 0.03) & (radius < 0.18)]) + 1e-12)
    )


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values)
    return {
        "n": len(values),
        "q10": float(np.quantile(arr, 0.1)),
        "median": float(np.median(arr)),
        "q90": float(np.quantile(arr, 0.9)),
    }


def main() -> None:
    rows = []
    for split, scanner_ids in (
        ("Valid", ("scanner03", "scanner04")),
        ("Test", ("scanner01", "scanner02")),
    ):
        with zipfile.ZipFile(ROOT / f"{split}.zip") as archive:
            for scanner in scanner_ids:
                clean_files = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith(f"{split}/clean/{scanner}_")
                    and name.endswith(".tif")
                )
                for clean_name in clean_files[:30]:
                    scan_name = clean_name.replace("/clean/", "/scan/")
                    clean, scan = (
                        _image(archive, clean_name),
                        _image(archive, scan_name),
                    )
                    for top in range(0, 1024, TILE):
                        for left in range(0, 1024, TILE):
                            a = clean[top : top + TILE, left : left + TILE]
                            b = scan[top : top + TILE, left : left + TILE]
                            al, bl = _luma(a), _luma(b)
                            # Gating is defined on the digital source only.
                            if not (0.15 < al.mean() < 0.9 and al.std() < 0.045):
                                continue
                            rows.append(
                                {
                                    "split": split,
                                    "scanner": scanner,
                                    "file": Path(clean_name).name,
                                    "x": left,
                                    "y": top,
                                    "clean_std": float(al.std()),
                                    "scan_std": float(bl.std()),
                                    "clean_hf_lf": _high_band(al),
                                    "scan_hf_lf": _high_band(bl),
                                    "clean_chroma_spread": float(a.std(axis=2).mean()),
                                    "scan_chroma_spread": float(b.std(axis=2).mean()),
                                }
                            )
    summary = {}
    for scanner in sorted({r["scanner"] for r in rows}):
        group = [r for r in rows if r["scanner"] == scanner]
        summary[scanner] = {
            feature: _quantiles([r[feature] for r in group])
            for feature in (
                "clean_std",
                "scan_std",
                "clean_hf_lf",
                "scan_hf_lf",
                "clean_chroma_spread",
                "scan_chroma_spread",
            )
        }
        summary[scanner]["pages_with_selected_tiles"] = len({r["file"] for r in group})
    result = {
        "selection": "first 30 files per scanner; non-overlap 128 tiles; clean luminance mean (0.15,0.9) and std <0.045",
        "summary": summary,
        "rows": rows,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
