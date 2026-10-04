"""Create a frozen 256x256 recapture control for Chimera B-Free evaluation.

This is a sensitivity control, not a reconstruction of the camera process.
Only ordinary recaptures are resized; the originals are already 256x256.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

from palimpsest.io.hashing import file_sha256 as sha256

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


EXPECTED_SOURCE_MANIFEST_SHA = (
    "39da9e9eac3bd2d133c53769f9c5a158c1769039c67560c0bbfc4660d09a7d24"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DATA_ROOT / "derived/chimera_paired",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_ROOT / "derived/chimera_recap256",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DATA_ROOT / "manifests/chimera_bfree_manifest.csv",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DATA_ROOT / "manifests/chimera_recap256_bfree_manifest.csv",
    )
    parser.add_argument(
        "--audit-json", type=Path, default=WORK_DIR / "chimera_recap256_audit.json"
    )
    args = parser.parse_args()

    if sha256(args.source_manifest) != EXPECTED_SOURCE_MANIFEST_SHA:
        raise RuntimeError("source manifest has changed")
    with args.source_manifest.open(newline="", encoding="utf-8-sig") as stream:
        original_rows = list(csv.DictReader(stream))
    rows = [
        row
        for row in original_rows
        if row["condition"] in ("recap_mac", "recap_monitor")
    ]
    if len(rows) != 2400:
        raise RuntimeError("expected exactly 2400 ordinary recaptures")
    args.output_root.mkdir(parents=True, exist_ok=True)
    produced = []
    with args.output_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["filename", "src", "label", "condition"]
        )
        writer.writeheader()
        for row in rows:
            source = args.source_root / row["filename"]
            destination = args.output_root / row["filename"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                if image.format != "PNG":
                    raise RuntimeError(f"unexpected format: {source}")
                reduced = image.convert("RGB").resize(
                    (256, 256), Image.Resampling.LANCZOS
                )
                reduced.save(destination, format="PNG")
            with Image.open(destination) as check:
                check.load()
                if check.size != (256, 256) or check.mode != "RGB":
                    raise RuntimeError(f"failed output decode: {destination}")
            writer.writerow(row)
            produced.append(
                {"filename": row["filename"], "sha256": sha256(destination)}
            )

    audit = {
        "purpose": "B-Free sensitivity control: published physical recaptures resized to source dimensions",
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA,
        "output_manifest": str(args.output_manifest),
        "output_manifest_sha256": sha256(args.output_manifest),
        "output_root": str(args.output_root),
        "method": "Pillow RGB conversion then LANCZOS downsample to 256x256, saved as PNG",
        "count": len(produced),
        "condition_counts": {
            condition: sum(row["condition"] == condition for row in rows)
            for condition in ("recap_mac", "recap_monitor")
        },
        "image_sha256": produced,
    }
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: audit[key]
                for key in ("count", "condition_counts", "output_manifest_sha256")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
