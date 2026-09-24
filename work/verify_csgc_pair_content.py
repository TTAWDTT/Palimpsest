"""Check that CSGC's same-number source/scan entries are content paired."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ARCHIVE = Path(r"E:\ai_image_origin_research\data\raw\csgc\CSGC_scan2400spi.zip")
OUT = Path("work/csgc2400_pair_content.json")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float32)
    b = b.ravel().astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


with zipfile.ZipFile(ARCHIVE) as archive:
    prior_binary = None
    first_binary = None
    same = []
    adjacent_wrong = []
    for item in range(1, 951):
        code = f"Scan{item:05d}.tif"
        with Image.open(io.BytesIO(archive.read(f"2400dpi_NEW/bin_exact_crop_4/{code}"))) as im:
            binary = np.asarray(im, dtype=np.uint8)
        with Image.open(io.BytesIO(archive.read(f"2400dpi_NEW/resize_exact_crop/{code}"))) as im:
            scan = np.asarray(im, dtype=np.uint8)
        same.append(corr(binary, scan))
        if prior_binary is not None:
            adjacent_wrong.append(corr(prior_binary, scan))
        else:
            first_binary = binary.copy()
        prior_binary = binary
    adjacent_wrong.append(corr(prior_binary, np.asarray(Image.open(io.BytesIO(
        archive.read("2400dpi_NEW/resize_exact_crop/Scan00001.tif"))), dtype=np.uint8)))

same = np.asarray(same)
wrong = np.asarray(adjacent_wrong)
report = {
    "n": int(len(same)),
    "same_id_pearson_quantiles": {str(q): float(np.quantile(same, q)) for q in (0, .05, .25, .5, .75, .95, 1)},
    "adjacent_wrong_id_pearson_quantiles": {str(q): float(np.quantile(wrong, q)) for q in (0, .05, .25, .5, .75, .95, 1)},
    "same_greater_than_wrong_fraction": float(np.mean(same > wrong)),
    "all_same_finite": bool(np.isfinite(same).all()),
}
OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
