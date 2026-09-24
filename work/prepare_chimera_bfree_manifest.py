"""Decode all source and genuine recapture images and sign an inference manifest."""

import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image


ROOT = Path("E:/ai_image_origin_research/data/derived/chimera_paired")
MANIFEST = Path("E:/ai_image_origin_research/data/manifests/chimera_bfree_manifest.csv")
IMAGE_AUDIT = Path("E:/ai_image_origin_research/data/manifests/chimera_paired_image_audit.csv")
SUMMARY = Path("work/chimera_extracted_audit.json")
CONDITIONS = ("stylegan2_orig", "recap_mac", "recap_monitor")
CLASSES = ("cat", "church", "horse")
LABELS = ("0_real", "1_fake")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    image_rows = []
    inference_rows = []
    for klass in CLASSES:
        for label_dir in LABELS:
            source_directory = ROOT / "stylegan2_orig" / klass / label_dir
            ids = sorted(path.name for path in source_directory.glob("*.png"))
            if len(ids) != 200:
                raise RuntimeError(f"expected 200 source images in {source_directory}, found {len(ids)}")
            for condition in CONDITIONS:
                folder = ROOT / condition / klass / label_dir
                actual = {path.name for path in folder.glob("*.png")}
                if actual != set(ids):
                    raise RuntimeError(f"filenames differ from source at {folder}")
            for filename in ids:
                source_id = f"{klass}/{label_dir}/{filename}"
                label = "FAKE" if label_dir == "1_fake" else "REAL"
                for condition in CONDITIONS:
                    relative = f"{condition}/{source_id}"
                    path = ROOT / relative
                    raw = path.read_bytes()
                    with Image.open(BytesIO(raw)) as image:
                        image.load()
                        width, height, mode = image.width, image.height, image.mode
                    if mode != "RGB":
                        raise RuntimeError(f"unexpected image mode {mode}: {relative}")
                    image_rows.append({
                        "filename": relative, "src": source_id,
                        "condition": condition, "label": label,
                        "width": width, "height": height,
                        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                    })
                    inference_rows.append({
                        "filename": relative, "src": source_id,
                        "label": label, "condition": condition,
                    })
    for destination, fields, rows in (
        (IMAGE_AUDIT, tuple(image_rows[0]), image_rows),
        (MANIFEST, tuple(inference_rows[0]), inference_rows),
    ):
        with destination.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    dimensions = {}
    for condition in CONDITIONS:
        pairs = sorted({(row["width"], row["height"]) for row in image_rows
                        if row["condition"] == condition})
        dimensions[condition] = pairs
    summary = {
        "archive_audit": "work/chimera_archive_audit.json",
        "dataset_root": str(ROOT),
        "source_groups": 1200,
        "decoded_images": len(image_rows),
        "count_by_condition": {condition: sum(row["condition"] == condition
                                              for row in image_rows)
                               for condition in CONDITIONS},
        "dimensions_by_condition": dimensions,
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "image_audit": str(IMAGE_AUDIT),
        "image_audit_sha256": sha256(IMAGE_AUDIT),
        "audit_passed": len(image_rows) == 3600,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
