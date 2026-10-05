"""Search the MD5-verified official CIFAR-10 archive for captured screen stimuli.

This is a content-retrieval probe. The display quadrilateral is manually set
from RGB frame 0 and must not be interpreted as calibrated screen geometry."""

from __future__ import annotations
import hashlib
import json
import numpy as np
from PIL import Image
from experiments.data_preparation.raw2event.prepare_download_cifar10_python import md5

from experiments.screen_capture.source_matching.raw2event.protocol import (
    ARCHIVE,
    AUDIT,
    OUT,
    VIEWS,
    SAMPLES,
    load_class,
    normalized_gray,
    query_from_photo,
)


def main() -> None:
    if (
        not AUDIT.exists()
        or json.loads(AUDIT.read_text(encoding="utf-8"))["md5"]
        != "c58f30108f718f92721af3b95e74349a"
        or ARCHIVE.stat().st_size != 170_498_071
        or md5(ARCHIVE) != "c58f30108f718f92721af3b95e74349a"
    ):
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
            top.append(
                {
                    "rank": rank,
                    "score_gray_zncc": float(scores[index]),
                    **locators[index],
                }
            )
        source = images[ranked[0]]
        source_hash = hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest()
        parts = prefix.split("_")
        expected_batch = f"data_batch_{parts[2]}"
        expected_row = int(parts[3])
        Image.fromarray(query, "RGB").resize((256, 256), Image.Resampling.NEAREST).save(
            VIEWS / f"{prefix}_query.png"
        )
        Image.fromarray(source, "RGB").save(VIEWS / f"{prefix}_source32.png")
        Image.fromarray(images[ranked[0]], "RGB").resize(
            (256, 256), Image.Resampling.NEAREST
        ).save(VIEWS / f"{prefix}_best.png")
        records[prefix] = {
            "label": config["label"],
            "manual_corners_tl_tr_br_bl": config["corners_tl_tr_br_bl"],
            "candidate_count": len(images),
            "top10": top,
            "top1_vs_top2_gap": float(scores[ranked[0]] - scores[ranked[1]]),
            "top1_rgb_pixel_sha256": source_hash,
            "top1_batch_row_matches_prefix": (
                top[0]["batch"] == expected_batch and top[0]["row"] == expected_row
            ),
        }
    report = {
        "scope": "content retrieval from RGB frame0 against official CIFAR-10 class candidates",
        "metric": "grayscale zero-mean normalized cross-correlation at 32x32",
        "corners_status": "manual, approximate, not device geometry calibration",
        "records": records,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
