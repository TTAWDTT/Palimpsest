"""Visualize selected source/RGB/RAW registration failures for diagnosis."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from work.audit_raw2event_probe import ROOT, extract_frame
from work.audit_raw2event_split_first_frames import load_originals
from work.probe_raw2event_source_to_raw import prepare


SPLIT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
RGB = Path("work/raw2event_content_registered_geometry.json")
RAW = Path("work/raw2event_raw_target_registration_diagnostic.json")
OUT = Path("work/raw2event_registration_diagnostic.png")
CLASSES = ("airplane", "bird", "cat", "deer", "ship")


def crop(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32),
                                           np.asarray([[0, 0], [127, 0], [127, 127], [0, 127]], dtype=np.float32))
    return cv2.warpPerspective(image, transform, (128, 128), flags=cv2.INTER_LINEAR)


def main() -> None:
    rows = [row for row in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
            if row["role"] in ("calibration", "development") and row["class_name"] in CLASSES]
    source = load_originals(rows)
    rg = {row["prefix"]: row for row in json.loads(RGB.read_text(encoding="utf-8"))["records"]}
    rw = {row["prefix"]: row for row in json.loads(RAW.read_text(encoding="utf-8"))["records"]}
    panels = []
    for row in rows:
        prefix = row["prefix"]
        rgb = extract_frame(ROOT / "frames_rgb" / f"{prefix}.mkv", 0, "rgb24", 3, "u1")
        raw = extract_frame(ROOT / "frames_raw" / f"{prefix}.mkv", 0, "gray16le", 1, "<u2")
        raw = cv2.GaussianBlur(raw.astype(np.float32), (0, 0), 1.5)
        raw = np.clip(raw / 600 * 255, 0, 255).astype(np.uint8)
        corners = np.asarray(rg[prefix]["refined_corners"], dtype=np.float32)
        prepared = prepare(prefix, None, source[prefix], corners)
        images = [cv2.resize(source[prefix], (128, 128), interpolation=cv2.INTER_NEAREST),
                  crop(rgb, corners),
                  np.repeat(crop(raw, prepared["raw_corners"])[..., None], 3, axis=2),
                  np.repeat(crop(raw, np.asarray(rw[prefix]["target_refined_raw_corners"]))[..., None], 3, axis=2)]
        strip = np.concatenate(images, axis=1)
        canvas = np.full((158, 512, 3), 255, dtype=np.uint8)
        canvas[30:, :] = strip
        label = f"{row['class_name']} {row['role'][:3]} RAW corr {rw[prefix]['raw_zncc_before']:.2f}->{rw[prefix]['raw_zncc_after']:.2f}"
        cv2.putText(canvas, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(canvas)
    grid = np.concatenate(panels, axis=0)
    cv2.imwrite(str(OUT), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(OUT)


if __name__ == "__main__":
    main()
