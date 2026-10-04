"""Audit observable encoding provenance of RR transformed test images."""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin


BASE = REPO_ROOT
ROOT = Path(r"E:\ai_image_origin_research\data\derived\rr_test")
MANIFEST = Path(r"E:\ai_image_origin_research\data\manifests\rr_test_files.csv")
OUTPUT = BASE / "work" / "rr_transformation_provenance_audit.json"


def fingerprint(qtables: dict[int, list[int]]) -> str:
    payload = b"".join(
        np.asarray(qtables[index], dtype=np.uint16).tobytes()
        for index in sorted(qtables)
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(Counter)
    profile_examples = {}
    for row in rows:
        condition = row["condition"]
        if condition == "original":
            continue
        with Image.open(ROOT / row["filename"]) as image:
            qtable_id = (
                fingerprint(image.quantization) if image.quantization else "none"
            )
            groups[condition]["qtables/" + qtable_id] += 1
            groups[condition][
                "subsampling/" + str(JpegImagePlugin.get_sampling(image))
            ] += 1
            groups[condition]["exif_present/" + str(bool(image.info.get("exif")))] += 1
            groups[condition][
                "icc_present/" + str(bool(image.info.get("icc_profile")))
            ] += 1
            groups[condition]["format/" + str(image.format)] += 1
            if qtable_id not in profile_examples:
                profile_examples[qtable_id] = row["filename"]
    profile = {}
    for condition, counts in groups.items():
        profile[condition] = {
            "images": sum(
                value for key, value in counts.items() if key.startswith("format/")
            ),
            "qtables": dict(
                (key.removeprefix("qtables/"), value)
                for key, value in counts.items()
                if key.startswith("qtables/")
            ),
            "subsampling": dict(
                (key.removeprefix("subsampling/"), value)
                for key, value in counts.items()
                if key.startswith("subsampling/")
            ),
            "exif_present": dict(
                (key.removeprefix("exif_present/"), value)
                for key, value in counts.items()
                if key.startswith("exif_present/")
            ),
            "icc_present": dict(
                (key.removeprefix("icc_present/"), value)
                for key, value in counts.items()
                if key.startswith("icc_present/")
            ),
        }
    standard = Image.new("RGB", (8, 8), "white")
    import io

    buffer = io.BytesIO()
    standard.save(buffer, format="JPEG", quality=95, subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as image:
        standard_q95_fingerprint = fingerprint(image.quantization)
    result = {
        "manifest": str(MANIFEST),
        "standard_pillow_q95_444_fingerprint": standard_q95_fingerprint,
        "conditions": profile,
        "qtable_examples": profile_examples,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "standard_q95": standard_q95_fingerprint,
                "conditions": {
                    key: {
                        field: value
                        for field, value in profile[key].items()
                        if field != "qtables"
                    }
                    | {
                        "qtables_top": sorted(
                            profile[key]["qtables"].items(), key=lambda pair: -pair[1]
                        )[:10]
                    }
                    for key in profile
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
