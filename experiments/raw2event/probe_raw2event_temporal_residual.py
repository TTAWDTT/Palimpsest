"""Explore temporal RAW differences in locally flat CFA planes.

The output is an *effective residual*, not a sensor read-noise estimate:
camera motion, display modulation, compression, illumination and rolling
exposure remain mixed. The script uses unsupervised flatness masks only.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("E:/ai_image_origin_research/data/raw/raw2event_probe/frames_raw")
PREFIXES = (
    "10000_automobile_5_1087_20251224_105416",
    "1000_airplane_1_9934_20251222_161953",
    "54380_truck_1_4340_20260121_043430",
)
OUT = Path("work/raw2event_temporal_residual_probe.json")
WIDTH, HEIGHT = 692, 520
THRESHOLDS = (2.0, 4.0, 8.0)
INTENSITY_EDGES = (0, 128, 256, 384, 512, 1024)
MAX_ABS_SECOND_DIFF = 512


def frames(path: Path):
    proc = subprocess.Popen(
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
    assert proc.stdout is not None
    try:
        while True:
            payload = proc.stdout.read(WIDTH * HEIGHT * 2)
            if not payload:
                break
            if len(payload) != WIDTH * HEIGHT * 2:
                raise RuntimeError("short decoded frame")
            yield np.frombuffer(payload, "<u2").reshape(HEIGHT, WIDTH).copy()
    finally:
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def main() -> None:
    results = []
    for prefix in PREFIXES:
        iterator = iter(frames(ROOT / f"{prefix}.mkv"))
        prev, current = next(iterator), next(iterator)
        buckets = {
            f"{threshold:g}": np.zeros(
                (4, len(INTENSITY_EDGES) - 1, MAX_ABS_SECOND_DIFF + 1), dtype=np.int64
            )
            for threshold in THRESHOLDS
        }
        triplets = 0
        total_frames = 2
        for nxt in iterator:
            total_frames += 1
            if triplets % 5 == 0:
                for parity, (y, x) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                    a = prev[y::2, x::2].astype(np.float32)
                    b = current[y::2, x::2].astype(np.float32)
                    c = nxt[y::2, x::2].astype(np.float32)
                    avg = (a + b + c) / 3
                    local_mean = cv2.GaussianBlur(avg, (5, 5), 0)
                    local_mean_sq = cv2.GaussianBlur(avg * avg, (5, 5), 0)
                    local_std = np.sqrt(np.maximum(0, local_mean_sq - local_mean**2))
                    # Trim frame margins and a small derivative from local motion.
                    second_abs = np.abs(a - 2 * b + c).astype(np.int32)
                    intensity = np.digitize(avg, INTENSITY_EDGES) - 1
                    for threshold in THRESHOLDS:
                        key = f"{threshold:g}"
                        eligible = local_std < threshold
                        eligible[:4] = eligible[-4:] = False
                        eligible[:, :4] = eligible[:, -4:] = False
                        for band in range(len(INTENSITY_EDGES) - 1):
                            mask = eligible & (intensity == band)
                            if not np.any(mask):
                                continue
                            hist = np.bincount(
                                np.clip(second_abs[mask], 0, MAX_ABS_SECOND_DIFF),
                                minlength=MAX_ABS_SECOND_DIFF + 1,
                            )
                            buckets[key][parity, band] += hist
            triplets += 1
            prev, current = current, nxt
        record = {
            "prefix": prefix,
            "frames_decoded": total_frames,
            "triplets_sampled": (triplets + 4) // 5,
            "thresholds": {},
        }
        for key, hist in buckets.items():
            records = []
            for parity in range(4):
                for band in range(len(INTENSITY_EDGES) - 1):
                    values = hist[parity, band]
                    count = int(values.sum())
                    median_abs = (
                        int(np.searchsorted(np.cumsum(values), (count + 1) // 2))
                        if count
                        else None
                    )
                    records.append(
                        {
                            "cfa_parity_yx": [parity // 2, parity % 2],
                            "intensity_interval": [
                                INTENSITY_EDGES[band],
                                INTENSITY_EDGES[band + 1],
                            ],
                            "n": count,
                            "median_abs_second_difference_counts": median_abs,
                            "effective_sigma_counts_if_iid": None
                            if median_abs is None
                            else float(median_abs / (0.67448975 * np.sqrt(6))),
                        }
                    )
            record["thresholds"][key] = records
        results.append(record)
        print(
            prefix,
            total_frames,
            [
                (key, sum(x["n"] for x in record["thresholds"][key]))
                for key in record["thresholds"]
            ],
            flush=True,
        )
    report = {
        "method": "RAW same-pixel second temporal difference; every fifth frame triplet; CFA-separated local 5x5 flatness",
        "qualification": "effective residual includes motion, display modulation and other temporal process; iid sigma is conditional interpretation only",
        "records": results,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
