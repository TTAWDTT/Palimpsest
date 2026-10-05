"""Released Raw2Event decoding conventions, without data access on import.

692x520 is the audited release frame shape, not a general camera assumption.
"""

from pathlib import Path

import cv2
import numpy as np

from palimpsest.paths import DATA_ROOT
from palimpsest.data.video import extract_frame as decode_frame

ROOT = DATA_ROOT / "raw/raw2event_probe"
WIDTH, HEIGHT = 692, 520


def extract_frame(
    path: Path, index: int, pix_fmt: str, channels: int, dtype: str
) -> np.ndarray:
    """Decode using the historical Raw2Event release dimensions."""
    return decode_frame(
        path, index, pix_fmt, channels, dtype, width=WIDTH, height=HEIGHT
    )


def detect_tag(gray_or_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    )
    corners, ids, _ = detector.detectMarkers(gray_or_rgb)
    if ids is None or len(ids) != 1:
        raise RuntimeError(f"expected exactly one 36h11 AprilTag; got {ids}")
    return corners[0].reshape(4, 2), int(ids[0, 0])
