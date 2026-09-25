"""Independently test the filename-to-CIFAR mapping on frozen captured sources.

Ranks the named source against all 6,000 originals of the same CIFAR class,
using only the first real ISP-RGB frame and the frozen Tag-based approximate
content quadrilateral. This is an identity check, not process calibration.
"""

import argparse
import csv
import io
import json
import pickle
import tarfile
import warnings
from pathlib import Path

import cv2
import numpy as np

from work.audit_raw2event_probe import ROOT, extract_frame
from work.fetch_cifar10_python import md5
from work.match_raw2event_cifar_source import normalized_gray
from work.audit_raw2event_split_first_frames import SOURCE_QUAD


SPLIT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
ARCHIVE = Path("E:/ai_image_origin_research/data/raw/cifar10_official/cifar-10-python.tar.gz")
GEOMETRY = Path("work/raw2event_process_split_first_frame_audit.json")
OUT = Path("work/raw2event_process_split_source_identity")
OFFICIAL_MD5 = "c58f30108f718f92721af3b95e74349a"


def load_candidates() -> dict[int, tuple[np.ndarray, list[tuple[str, int]]]]:
    blocks = {label: [] for label in range(10)}
    keys = {label: [] for label in range(10)}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
            with archive.extractfile(f"cifar-10-batches-py/{name}") as stream:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed.*")
                    batch = pickle.load(io.BytesIO(stream.read()), encoding="bytes")
            images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.asarray(batch[b"labels"], dtype=np.uint8)
            for label in range(10):
                indexes = np.flatnonzero(labels == label)
                blocks[label].append(images[indexes])
                keys[label].extend((name, int(index)) for index in indexes)
    return {label: (normalized_gray(np.concatenate(blocks[label])), keys[label]) for label in range(10)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=("fixed_quad", "tag_similarity", "source_rgb_refined"),
                        default="tag_similarity")
    args = parser.parse_args()
    if ARCHIVE.stat().st_size != 170_498_071 or md5(ARCHIVE) != OFFICIAL_MD5:
        raise RuntimeError("official CIFAR-10 archive failed verification")
    split = [row for row in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
             if row["role"] in ("calibration", "development")]
    audit = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    geometry = {row["prefix"]: row for row in audit["rows"]}
    if len(split) != 20 or audit["count"] != 20 or set(geometry) != {row["prefix"] for row in split}:
        raise RuntimeError("complete frozen split and geometry audit required")
    refined = None
    if args.geometry == "source_rgb_refined":
        refined_report = json.loads(Path("work/raw2event_content_registered_geometry.json").read_text(encoding="utf-8"))
        refined = {row["prefix"]: row for row in refined_report["records"]}
        if refined_report["n"] != 20 or set(refined) != set(geometry):
            raise RuntimeError("refined geometry mismatch")
    candidates = load_candidates()
    records = []
    for row in split:
        prefix = row["prefix"]
        rgb = extract_frame(ROOT / "frames_rgb" / f"{prefix}.mkv", 0, "rgb24", 3, "u1")
        corners = np.asarray(SOURCE_QUAD if args.geometry == "fixed_quad" else
                             geometry[prefix]["tag_similarity_content_corners"] if refined is None else
                             refined[prefix]["refined_corners"], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(corners, np.asarray([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.float32))
        query = cv2.warpPerspective(rgb, transform, (32, 32))
        label = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck").index(row["class_name"])
        features, keys = candidates[label]
        scores = features @ normalized_gray(query[None])[0]
        target = (row["cifar_batch"], int(row["row"]))
        try:
            target_index = keys.index(target)
        except ValueError as exc:
            raise RuntimeError(f"named source absent from its CIFAR class: {prefix}") from exc
        order = np.argsort(scores)[::-1]
        rank = int(np.flatnonzero(order == target_index)[0]) + 1
        runner_up = next(int(i) for i in order if i != target_index)
        records.append({"prefix": prefix, "role": row["role"], "class_name": row["class_name"],
                        "named_source_rank_among_class": rank,
                        "named_source_zncc": float(scores[target_index]),
                        "best_other_zncc": float(scores[runner_up]),
                        "best_other_batch_row": [keys[runner_up][0], keys[runner_up][1]]})
        print(f"{len(records)}/20 {prefix}: named rank {rank}", flush=True)
    report = {"scope": "frozen 20 captured first frames, 6,000 same-class official CIFAR originals each",
              "geometry": args.geometry,
              "method": "grayscale 32x32 ZNCC; source_rgb_refined geometry directly optimizes the named source and is not independent identity validation",
              "cifar_archive_md5": OFFICIAL_MD5, "n": len(records),
              "named_top1": sum(record["named_source_rank_among_class"] == 1 for record in records),
              "records": records}
    OUT.with_name(OUT.name + f"_{args.geometry}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": report["n"], "named_top1": report["named_top1"]}))


if __name__ == "__main__":
    main()
