"""Compare one independently packaged Raw2Event RAW clip with direct views."""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, WORK_DIR

import json
import subprocess
import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PREFIX = "10000_automobile_5_1087_20251224_105416"
ROOT = DATA_ROOT / "raw/raw2event_probe"


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,nb_read_frames,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def pts(path: Path) -> list[str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        entry["best_effort_timestamp_time"]
        for entry in json.loads(result.stdout)["frames"]
    ]


def frame(path: Path, index: int, width: int, height: int) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    arr = np.frombuffer(result.stdout, dtype="<u2")
    if arr.size != width * height:
        raise RuntimeError(f"bad decoded frame {path} {index}: {arr.size}")
    return arr.reshape(height, width).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--out", type=Path, default=WORK_DIR / "raw2event_tar_vs_direct_audit.json"
    )
    args = parser.parse_args()
    prefix = args.prefix
    tar_path = ROOT / "raw_tar_member" / f"raw_frames_{prefix}_raw_10bit.mkv"
    direct_path = ROOT / "frames_raw" / f"{prefix}.mkv"
    tar_probe, direct_probe = probe(tar_path), probe(direct_path)
    ts, ds = tar_probe["streams"][0], direct_probe["streams"][0]
    tw, th = ts["width"], ts["height"]
    dw, dh = ds["width"], ds["height"]
    if tw > dw or th > dh:
        raise RuntimeError("TAR RAW unexpectedly larger")
    center_x, center_y = (dw - tw) // 2, (dh - th) // 2
    samples = []
    for index in (0, 80, 160, 240, 316):
        a = frame(tar_path, index, tw, th)
        b = frame(direct_path, index, dw, dh)
        centered = b[center_y : center_y + th, center_x : center_x + tw]
        dif = a.astype(np.int32) - centered.astype(np.int32)
        # More general translation/photometric check if simple cropping fails.
        a8 = np.clip(a / 4, 0, 255).astype(np.uint8)
        b8 = np.clip(b / 4, 0, 255).astype(np.uint8)
        response = cv2.matchTemplate(b8, a8, cv2.TM_CCOEFF_NORMED)
        _, best_score, _, (best_x, best_y) = cv2.minMaxLoc(response)
        best = b[best_y : best_y + th, best_x : best_x + tw]
        best_dif = a.astype(np.int32) - best.astype(np.int32)
        samples.append(
            {
                "frame": index,
                "tar_range": [int(a.min()), int(a.max())],
                "direct_range": [int(b.min()), int(b.max())],
                "center_crop_offset_xy": [center_x, center_y],
                "center_exact_fraction": float(np.mean(dif == 0)),
                "center_mae": float(np.mean(np.abs(dif))),
                "center_pearson": float(np.corrcoef(a.ravel(), centered.ravel())[0, 1]),
                "best_translation_xy": [best_x, best_y],
                "best_ccoeff": float(best_score),
                "best_exact_fraction": float(np.mean(best_dif == 0)),
                "best_mae": float(np.mean(np.abs(best_dif))),
                "best_pearson": float(np.corrcoef(a.ravel(), best.ravel())[0, 1]),
                "tar_parity_means": [
                    [float(np.mean(a[y::2, x::2])) for x in (0, 1)] for y in (0, 1)
                ],
            }
        )
    candidate_offsets = {tuple(item["best_translation_xy"]) for item in samples}
    if len(candidate_offsets) != 1:
        raise RuntimeError(
            f"sampled frames do not have a common offset: {candidate_offsets}"
        )
    off_x, off_y = next(iter(candidate_offsets))

    def decoder(path: Path) -> subprocess.Popen:
        return subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray16le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    tar_decoder, direct_decoder = decoder(tar_path), decoder(direct_path)
    n_frames = 0
    unequal_frames = 0
    unequal_pixels = 0
    first_unequal_frame = None
    assert tar_decoder.stdout is not None and direct_decoder.stdout is not None
    while True:
        tar_bytes = tar_decoder.stdout.read(tw * th * 2)
        direct_bytes = direct_decoder.stdout.read(dw * dh * 2)
        if not tar_bytes and not direct_bytes:
            break
        if len(tar_bytes) != tw * th * 2 or len(direct_bytes) != dw * dh * 2:
            raise RuntimeError("decoded stream length mismatch")
        a = np.frombuffer(tar_bytes, "<u2").reshape(th, tw)
        b = np.frombuffer(direct_bytes, "<u2").reshape(dh, dw)
        bad = int(np.count_nonzero(a != b[off_y : off_y + th, off_x : off_x + tw]))
        if bad:
            unequal_frames += 1
            unequal_pixels += bad
            if first_unequal_frame is None:
                first_unequal_frame = n_frames
        n_frames += 1
    if tar_decoder.wait() or direct_decoder.wait():
        raise RuntimeError("ffmpeg decoder failed")
    if n_frames != int(ts["nb_read_frames"]) or n_frames != int(ds["nb_read_frames"]):
        raise RuntimeError("decoded frame counts differ from probe")
    tar_pts, direct_pts = pts(tar_path), pts(direct_path)
    report = {
        "prefix": prefix,
        "tar": tar_probe,
        "direct": direct_probe,
        "dimension_difference_wh": [dw - tw, dh - th],
        "samples": samples,
        "timestamp_frames_exact": len(tar_pts) == len(direct_pts) == n_frames
        and tar_pts == direct_pts,
        "pts_first_last": [tar_pts[0], tar_pts[-1]],
        "all_frames_crop_check": {
            "offset_xy": [off_x, off_y],
            "frames_checked": n_frames,
            "pixels_checked": n_frames * tw * th,
            "unequal_frames": unequal_frames,
            "unequal_pixels": unequal_pixels,
            "first_unequal_frame": first_unequal_frame,
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
