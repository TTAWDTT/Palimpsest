"""Decode video timestamps and single frames through ffprobe/ffmpeg.

The caller supplies the expected decoded dimensions; no resize is performed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def ffprobe_frames(path: Path) -> list[float]:
    proc = subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(proc.stdout)["frames"]
    return [float(item["best_effort_timestamp_time"]) for item in values]


def extract_frame(
    path: Path,
    index: int,
    pix_fmt: str,
    channels: int,
    dtype: str,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    proc = subprocess.run(
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
            pix_fmt,
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    arr = np.frombuffer(proc.stdout, dtype=dtype)
    expected = width * height * channels
    if arr.size != expected:
        raise RuntimeError(
            f"bad decoded frame {index}: {arr.size} samples, expected {expected}"
        )
    shape = (height, width) if channels == 1 else (height, width, channels)
    return arr.reshape(shape).copy()
