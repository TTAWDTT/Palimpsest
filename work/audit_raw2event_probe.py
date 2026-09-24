"""Read-only structural and pixel audit of one official Raw2Event prefix.

No pickle or vendor-specific decoding is used. The 38-byte sidecar record
layout is inferred empirically from this single file, not an official schema.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


ROOT = Path("E:/ai_image_origin_research/data/raw/raw2event_probe")
DEFAULT_PREFIX = "10000_automobile_5_1087_20251224_105416"
WIDTH, HEIGHT = 692, 520
INDICES = (0, 80, 160, 240, 316)


def ffprobe_frames(path: Path) -> list[float]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    values = json.loads(proc.stdout)["frames"]
    return [float(item["best_effort_timestamp_time"]) for item in values]


def extract_frame(path: Path, index: int, pix_fmt: str, channels: int, dtype: str) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})",
         "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"],
        check=True, capture_output=True,
    )
    arr = np.frombuffer(proc.stdout, dtype=dtype)
    expected = WIDTH * HEIGHT * channels
    if arr.size != expected:
        raise RuntimeError(f"bad decoded frame {index}: {arr.size} samples, expected {expected}")
    shape = (HEIGHT, WIDTH) if channels == 1 else (HEIGHT, WIDTH, channels)
    return arr.reshape(shape).copy()


def sidecar_audit(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) % 38:
        raise RuntimeError("sidecar does not match observed 38-byte stride")
    values = []
    times = []
    suffixes = []
    for offset in range(0, len(data), 38):
        record = data[offset:offset + 38]
        value = struct.unpack("<d", record[:8])[0]
        date_text = record[8:34].decode("ascii")
        if not re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{6}", date_text):
            raise RuntimeError(f"invalid date at offset {offset}: {date_text}")
        values.append(value)
        times.append(datetime.fromisoformat(date_text))
        suffixes.append(record[34:38].hex())
    diffs = np.diff(values)
    date_diffs = np.diff(np.array(times, dtype="datetime64[us]")).astype("timedelta64[us]").astype(np.int64)
    return {
        "record_count": len(values), "empirical_record_bytes": 38,
        "numeric_first_last": [values[0], values[-1]],
        "numeric_diffs_median": float(np.median(diffs)),
        "numeric_diffs_min_max": [float(np.min(diffs)), float(np.max(diffs))],
        "calendar_first_last": [times[0].isoformat(), times[-1].isoformat()],
        "calendar_diffs_us_median": float(np.median(date_diffs)),
        "calendar_diffs_us_min_max": [int(np.min(date_diffs)), int(np.max(date_diffs))],
        "numeric_monotonic": bool(np.all(diffs > 0)),
        "calendar_monotonic": bool(np.all(date_diffs > 0)),
        "suffix_counts": {x: suffixes.count(x) for x in sorted(set(suffixes))},
        "observed_exposure_gain_fields": False,
        "schema_status": "empirical, single-file only; do not treat 38 bytes as official schema",
    }


def sample_stats(raw: np.ndarray, rgb: np.ndarray) -> dict:
    bits_or = int(np.bitwise_or.reduce(raw.ravel()))
    # A whole-frame comparison cannot establish a registered RAW/RGB pair.
    # Use only scalar summaries and mosaic parity statistics here.
    return {
        "raw_min_max": [int(raw.min()), int(raw.max())],
        "raw_p01_p50_p99": [float(x) for x in np.quantile(raw, [0.01, 0.5, 0.99])],
        "raw_aggregate_bit_or": bits_or,
        "raw_bit_length": bits_or.bit_length(),
        "raw_nonzero_fraction": float(np.mean(raw != 0)),
        "raw_parity_means": [[float(np.mean(raw[y::2, x::2])) for x in (0, 1)] for y in (0, 1)],
        "rgb_channel_means": [float(x) for x in rgb.mean(axis=(0, 1))],
        "rgb_min_max": [int(rgb.min()), int(rgb.max())],
    }


def detect_tag(gray_or_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    corners, ids, _ = detector.detectMarkers(gray_or_rgb)
    if ids is None or len(ids) != 1:
        raise RuntimeError(f"expected exactly one 36h11 AprilTag; got {ids}")
    return corners[0].reshape(4, 2), int(ids[0, 0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--out-dir", type=Path, default=Path("work/raw2event_probe"))
    parser.add_argument("--report", type=Path, default=Path("work/raw2event_probe_pixel_audit.json"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "frames_raw" / f"{args.prefix}.mkv"
    rgb_path = ROOT / "frames_rgb" / f"{args.prefix}.mkv"
    metadata_path = ROOT / "meta_raw" / f"{args.prefix}.dat"
    raw_pts, rgb_pts = ffprobe_frames(raw_path), ffprobe_frames(rgb_path)
    if len(raw_pts) != len(rgb_pts):
        raise RuntimeError("RAW and RGB decoded frame counts differ")
    pts_delta = np.array(raw_pts) - np.array(rgb_pts)
    report = {
        "scope": "one official prefix; format and pixel probe",
        "prefix": args.prefix,
        "raw_frames": len(raw_pts), "rgb_frames": len(rgb_pts),
        "pts_exact_matches": int(np.count_nonzero(pts_delta == 0)),
        "pts_delta_seconds_min_max": [float(pts_delta.min()), float(pts_delta.max())],
        "raw_pts_first_last": [raw_pts[0], raw_pts[-1]],
        "rgb_pts_first_last": [rgb_pts[0], rgb_pts[-1]],
        "sidecar": sidecar_audit(metadata_path),
        "samples": {},
    }
    for index in INDICES:
        raw = extract_frame(raw_path, index, "gray16le", 1, "<u2")
        rgb = extract_frame(rgb_path, index, "rgb24", 3, "u1")
        # Store viewable data, with fixed 10-bit scale for RAW. These are only
        # visual audit aids; raw PNGs are not converted sensor measurements.
        Image.fromarray(rgb, "RGB").save(args.out_dir / f"rgb_{index:03d}.png")
        Image.fromarray(np.minimum(raw.astype(np.float32) / 1023.0 * 255, 255).astype(np.uint8), "L").save(
            args.out_dir / f"raw_view_{index:03d}.png")
        raw_view = np.minimum(raw.astype(np.float32) / 1023.0 * 255, 255).astype(np.uint8)
        raw_corners, raw_id = detect_tag(raw_view)
        rgb_corners, rgb_id = detect_tag(rgb)
        if raw_id != rgb_id:
            raise RuntimeError(f"tag ID mismatch at frame {index}")
        raw_edge = np.linalg.norm(np.roll(raw_corners, -1, axis=0) - raw_corners, axis=1).mean()
        rgb_edge = np.linalg.norm(np.roll(rgb_corners, -1, axis=0) - rgb_corners, axis=1).mean()
        transform = cv2.getPerspectiveTransform(raw_corners, rgb_corners)
        registered = cv2.warpPerspective(raw_view, transform, (WIDTH, HEIGHT))
        Image.fromarray(registered, "L").save(args.out_dir / f"raw_tag_registered_{index:03d}.png")
        valid = cv2.warpPerspective(np.ones_like(raw_view), transform, (WIDTH, HEIGHT)) == 1
        valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        rgb_luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        smooth = lambda arr: cv2.GaussianBlur(arr, (7, 7), 1.5).astype(np.float64)
        raw_smooth, reg_smooth, rgb_smooth = map(smooth, (raw_view, registered, rgb_luma))
        stats = sample_stats(raw, rgb)
        stats["tag"] = {
            "id": raw_id,
            "raw_corners": raw_corners.tolist(),
            "rgb_corners": rgb_corners.tolist(),
            "raw_edge_px_mean": float(raw_edge),
            "rgb_edge_px_mean": float(rgb_edge),
            "rgb_to_raw_tag_edge_ratio": float(raw_edge / rgb_edge),
            "raw_to_rgb_tag_homography": transform.tolist(),
            "geometry_note": "AprilTag plane maps one image; sensor RAW/ISP RGB are not pixel-registered in released files",
        }
        stats["luma_registration_probe"] = {
            "valid_fraction": float(valid.mean()),
            "pearson_unregistered": float(np.corrcoef(raw_smooth[valid], rgb_smooth[valid])[0, 1]),
            "pearson_tag_registered": float(np.corrcoef(reg_smooth[valid], rgb_smooth[valid])[0, 1]),
            "method": "7x7 Gaussian sigma1.5; common eroded warped-valid mask; diagnostic, not ISP fidelity",
        }
        report["samples"][str(index)] = stats
    first_h = np.asarray(report["samples"][str(INDICES[0])]["tag"]["raw_to_rgb_tag_homography"])
    for index in INDICES:
        tag = report["samples"][str(index)]["tag"]
        raw_corners = np.asarray(tag["raw_corners"], dtype=np.float32)[None, :, :]
        predicted = cv2.perspectiveTransform(raw_corners, first_h)[0]
        actual = np.asarray(tag["rgb_corners"])
        tag["first_frame_h_mean_corner_error_px"] = float(np.linalg.norm(predicted - actual, axis=1).mean())
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
