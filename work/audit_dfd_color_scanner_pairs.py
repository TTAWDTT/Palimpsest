"""Check image-content pairing for DFD's six exceptional full-page PNGs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


METADATA = Path("work/dfd_color_non_tiff_fullpages.json")
OUT = Path("work/dfd_color_scanner_pairs.json")


def preview(path: str) -> np.ndarray:
    with Image.open(path) as image:
        image.thumbnail((1000, 1400))
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def match(first: str, second: str) -> dict:
    source = preview(first)
    target = preview(second)
    sift = cv2.SIFT_create(nfeatures=2000)
    point_a, descriptor_a = sift.detectAndCompute(source, None)
    point_b, descriptor_b = sift.detectAndCompute(target, None)
    if descriptor_a is None or descriptor_b is None:
        raise ValueError("no SIFT features")
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptor_a, descriptor_b, k=2)
    good = [best for best, runner in candidates if best.distance < .75 * runner.distance]
    if len(good) < 12:
        raise ValueError(f"insufficient paired features: {len(good)}")
    camera_a = np.float32([point_a[m.queryIdx].pt for m in good])
    camera_b = np.float32([point_b[m.trainIdx].pt for m in good])
    transform, inliers = cv2.findHomography(camera_a, camera_b, cv2.RANSAC,
                                            ransacReprojThreshold=4)
    if transform is None:
        raise ValueError("homography failed")
    return {"first_preview_shape": list(source.shape),
            "second_preview_shape": list(target.shape),
            "ratio_matches": len(good), "ransac_inliers": int(inliers.sum()),
            "inlier_fraction": float(inliers.mean()),
            "homography_preview_coords": transform.tolist()}


def main() -> None:
    records = json.loads(METADATA.read_text(encoding="utf-8"))["records"]
    pairs = []
    for number in (1, 2, 3):
        first = next(row for row in records
                     if f"_{number}_a_Scanner1.png" in row["archive_member"])
        second = next(row for row in records
                      if f"_{number}_ a_Scanner2.png" in row["archive_member"])
        pairs.append({"chart_number": number,
                      "scanner1_member": first["archive_member"],
                      "scanner2_member": second["archive_member"],
                      **match(first["path"], second["path"])})
    report = {"scope": "low-resolution content registration, not proof of same physical sheet or scan settings",
              "pairs": pairs}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": [(p["chart_number"], p["ratio_matches"], p["ransac_inliers"])
                                for p in pairs]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
