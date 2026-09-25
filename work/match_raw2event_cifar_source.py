"""Search the MD5-verified official CIFAR-10 archive for captured screen stimuli.

This is a content-retrieval probe. The display quadrilateral is manually set
from RGB frame 0 and must not be interpreted as calibrated screen geometry.
"""

from __future__ import annotations

import io
import hashlib
import json
import pickle
import tarfile
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image

from work.fetch_cifar10_python import md5


ARCHIVE = Path("E:/ai_image_origin_research/data/raw/cifar10_official/cifar-10-python.tar.gz")
AUDIT = Path("work/cifar10_python_download_audit.json")
OUT = Path("work/raw2event_cifar_source_match.json")
VIEWS = Path("work/raw2event_cifar_matches")
SAMPLES = {
    "10000_automobile_5_1087_20251224_105416": {
        "label": 1, "rgb": Path("work/raw2event_probe/rgb_000.png"),
        "corners_tl_tr_br_bl": [[237, 180], [409, 195], [397, 370], [219, 353]],
    },
    "1000_airplane_1_9934_20251222_161953": {
        "label": 0, "rgb": Path("work/raw2event_probe_airplane/rgb_000.png"),
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
                    warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed.*")
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
    gray = np.tensordot(images.astype(np.float32) / 255, [0.299, 0.587, 0.114], axes=([-1], [0]))
    flat = gray.reshape(len(images), -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    return centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9)


def query_from_photo(rgb_path: Path, corners: list[list[int]]) -> np.ndarray:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    h = cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32),
                                    np.asarray([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.float32))
    return cv2.warpPerspective(rgb, h, (32, 32)).astype(np.uint8)


def main() -> None:
    if (not AUDIT.exists() or json.loads(AUDIT.read_text(encoding="utf-8"))["md5"] != "c58f30108f718f92721af3b95e74349a"
            or ARCHIVE.stat().st_size != 170_498_071 or md5(ARCHIVE) != "c58f30108f718f92721af3b95e74349a"):
        raise RuntimeError("official CIFAR-10 archive length/MD5 verification failed")
    VIEWS.mkdir(parents=True, exist_ok=True)
    by_label = {label: load_class(label) for label in (0, 1)}
    records = {}
    for prefix, config in SAMPLES.items():
        images, locators = by_label[config["label"]]
        query = query_from_photo(config["rgb"], config["corners_tl_tr_br_bl"])
        scores = normalized_gray(images) @ normalized_gray(query[None])[0]
        ranked = np.argsort(scores)[::-1][:10]
        top = []
        for rank, index in enumerate(ranked, start=1):
            top.append({"rank": rank, "score_gray_zncc": float(scores[index]), **locators[index]})
        source = images[ranked[0]]
        source_hash = hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest()
        parts = prefix.split("_")
        expected_batch = f"data_batch_{parts[2]}"
        expected_row = int(parts[3])
        Image.fromarray(query, "RGB").resize((256, 256), Image.Resampling.NEAREST).save(VIEWS / f"{prefix}_query.png")
        Image.fromarray(source, "RGB").save(VIEWS / f"{prefix}_source32.png")
        Image.fromarray(images[ranked[0]], "RGB").resize((256, 256), Image.Resampling.NEAREST).save(
            VIEWS / f"{prefix}_best.png")
        records[prefix] = {"label": config["label"], "manual_corners_tl_tr_br_bl": config["corners_tl_tr_br_bl"],
                           "candidate_count": len(images), "top10": top,
                           "top1_vs_top2_gap": float(scores[ranked[0]] - scores[ranked[1]]),
                           "top1_rgb_pixel_sha256": source_hash,
                           "top1_batch_row_matches_prefix": (top[0]["batch"] == expected_batch
                                                              and top[0]["row"] == expected_row)}
    report = {"scope": "content retrieval from RGB frame0 against official CIFAR-10 class candidates",
              "metric": "grayscale zero-mean normalized cross-correlation at 32x32",
              "corners_status": "manual, approximate, not device geometry calibration",
              "records": records}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
