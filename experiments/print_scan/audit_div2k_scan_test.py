"""Audit public DIV2K-SCAN iPhone XR test archive and three source probes."""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


RAW = DATA_ROOT / "raw/div2k_scan"
PROBE = DATA_ROOT / "derived/div2k_scan_original_probe"
OUTPUT = WORK_DIR / "div2k_scan_test_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _source_3_2(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    width, height = image.size
    wanted_height = min(height, round(width * 2 / 3))
    wanted_width = min(width, round(height * 3 / 2))
    x0, y0 = (width - wanted_width) // 2, (height - wanted_height) // 2
    crop = image.crop((x0, y0, x0 + wanted_width, y0 + wanted_height))
    return np.asarray(crop.resize(size, Image.Resampling.BICUBIC).convert("RGB"))


def _correspondence(source: np.ndarray, capture: np.ndarray) -> dict:
    # SIFT and a geometric inlier test establish content pairing; photometry
    # can differ substantially across print, lighting and phone ISP.
    a = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(capture, cv2.COLOR_RGB2GRAY)
    a = cv2.resize(a, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=2500)
    key_a, des_a = sift.detectAndCompute(a, None)
    key_b, des_b = sift.detectAndCompute(b, None)
    if des_a is None or des_b is None:
        return {"matches": 0, "inliers": 0}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in candidates if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return {"matches": len(good), "inliers": 0}
    points_a = np.float32([key_a[m.queryIdx].pt for m in good])
    points_b = np.float32([key_b[m.trainIdx].pt for m in good])
    transform, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 3.0)
    if transform is None:
        return {"matches": len(good), "inliers": 0}
    return {
        "matches": len(good),
        "inliers": int(mask.sum()),
        "inlier_fraction": float(mask.mean()),
        "homography_half_resolution": transform.tolist(),
    }


def main() -> None:
    archive_path = RAW / "test_xr.zip"
    dimension_counts = Counter()
    modes = Counter()
    metadata_keys = Counter()
    samples = []
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(
            info.filename
            for info in archive.infolist()
            if info.filename.endswith(".png")
        )
        expected = [f"xr/{i:04d}.png" for i in range(801, 901)]
        if names != expected or archive.testzip() is not None:
            raise ValueError(
                "test archive contents or CRC do not match 0801-0900 expected set"
            )
        for name in names:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                dimension_counts[str(image.size)] += 1
                modes[image.mode] += 1
                metadata_keys.update(image.info.keys())
        for number in (801, 802, 803):
            image_id = f"{number:04d}"
            with Image.open(PROBE / f"{image_id}.png") as original:
                with Image.open(io.BytesIO(archive.read(f"xr/{image_id}.png"))) as shot:
                    capture = np.asarray(shot.convert("RGB"))
                    source = _source_3_2(original, shot.size)
            sample = {
                "id": image_id,
                "digital_source_sha256": _sha256(PROBE / f"{image_id}.png"),
                "source_rgb_mean": source.mean((0, 1)).tolist(),
                "camera_publication_rgb_mean": capture.mean((0, 1)).tolist(),
                "source_luma_mean": float(
                    (source * [0.2126, 0.7152, 0.0722]).sum(2).mean() / 255
                ),
                "camera_publication_luma_mean": float(
                    (capture * [0.2126, 0.7152, 0.0722]).sum(2).mean() / 255
                ),
                "correspondence": _correspondence(source, capture),
            }
            samples.append(sample)
    result = {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256(archive_path),
        "files_expected": 100,
        "crc_all_passed": True,
        "dimensions": dict(dimension_counts),
        "modes": dict(modes),
        "pil_metadata_key_counts": dict(metadata_keys),
        "original_probe_count": len(samples),
        "probes": samples,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "probes"}, indent=2
        )
    )
    for sample in samples:
        print(
            sample["id"],
            sample["correspondence"]["inliers"],
            round(sample["source_luma_mean"], 3),
            round(sample["camera_publication_luma_mean"], 3),
        )


if __name__ == "__main__":
    main()
