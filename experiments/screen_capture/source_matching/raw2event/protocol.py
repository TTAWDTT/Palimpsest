"""Search the MD5-verified official CIFAR-10 archive for captured screen stimuli.

This is a content-retrieval probe. The display quadrilateral is manually set
from RGB frame 0 and must not be interpreted as calibrated screen geometry.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, WORK_DIR

import io
import pickle
import tarfile
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image


ARCHIVE = DATA_ROOT / "raw/cifar10_official/cifar-10-python.tar.gz"
AUDIT = WORK_DIR / "cifar10_python_download_audit.json"
OUT = WORK_DIR / "raw2event_cifar_source_match.json"
VIEWS = WORK_DIR / "raw2event_cifar_matches"
SAMPLES = {
    "10000_automobile_5_1087_20251224_105416": {
        "label": 1,
        "rgb": WORK_DIR / "raw2event_probe/rgb_000.png",
        "corners_tl_tr_br_bl": [[237, 180], [409, 195], [397, 370], [219, 353]],
    },
    "1000_airplane_1_9934_20251222_161953": {
        "label": 0,
        "rgb": WORK_DIR / "raw2event_probe_airplane/rgb_000.png",
        "corners_tl_tr_br_bl": [[237, 180], [409, 195], [397, 370], [219, 353]],
    },
}


def load_class(label: int) -> tuple[np.ndarray, list[dict]]:
    images, locators = [], []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for batch in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
            member = archive.getmember(f"cifar-10-batches-py/{batch}")
            with archive.extractfile(member) as stream:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=r"dtype\(\): align should be passed.*"
                    )
                    payload = pickle.load(io.BytesIO(stream.read()), encoding="bytes")
            labels = np.asarray(payload[b"labels"], dtype=np.int32)
            data = np.asarray(payload[b"data"], dtype=np.uint8)
            indices = np.flatnonzero(labels == label)
            images.append(data[indices].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            locators.extend({"batch": batch, "row": int(index)} for index in indices)
    result = np.concatenate(images, axis=0)
    if result.shape != (6000, 32, 32, 3):
        raise RuntimeError(f"unexpected official class shape: {result.shape}")
    return result, locators


def normalized_gray(images: np.ndarray) -> np.ndarray:
    gray = np.tensordot(
        images.astype(np.float32) / 255, [0.299, 0.587, 0.114], axes=([-1], [0])
    )
    flat = gray.reshape(len(images), -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    return centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9)


def query_from_photo(rgb_path: Path, corners: list[list[int]]) -> np.ndarray:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    h = cv2.getPerspectiveTransform(
        np.asarray(corners, dtype=np.float32),
        np.asarray([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.float32),
    )
    return cv2.warpPerspective(rgb, h, (32, 32)).astype(np.uint8)
