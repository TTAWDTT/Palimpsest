"""Materialize the frozen geometry/color/blur proxy on development sources.

This is a published-output composite control, not a calibrated physical screen,
optics, RAW and ISP simulator. It uses no development recapture pixels to render.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from experiments.chimera.evaluate_chimera_composite_photometry import (
    ROOT,
    design,
    load_rgb,
)


SPLIT = Path(
    "E:/ai_image_origin_research/data/manifests/chimera_simulation_source_split.csv"
)
GEOMETRY = Path("work/chimera_fixed_publication_geometry.json")
PHOTOMETRY = Path("work/chimera_composite_photometry.json")
BLUR = Path("work/chimera_effective_blur_proxy.json")
DEFAULT_ROOT = Path(
    "E:/ai_image_origin_research/data/derived/chimera_composite_sim_dev256"
)
DEFAULT_MANIFEST = Path(
    "E:/ai_image_origin_research/data/manifests/chimera_composite_sim_dev256_manifest.csv"
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def render(
    source: str, condition: str, geometry: dict, photometry: dict, blur: dict
) -> np.ndarray:
    original = load_rgb(ROOT / "stylegan2_orig" / source)
    coefficient = np.asarray(
        photometry["conditions"][condition]["rgb3_plus_bias_coefficient"],
        dtype=np.float64,
    )
    aligned_prediction = np.clip(
        (design(original) @ coefficient).reshape(256, 256, 3), 0, 1
    )
    sigma = float(blur["conditions"][condition]["selected_sigma_output_pixels"])
    if sigma > 0:
        aligned_prediction = cv2.GaussianBlur(
            aligned_prediction,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
    calibration_warp = np.asarray(
        geometry["conditions"][condition]["median_calibration_affine"], dtype=np.float32
    )
    recap_grid = cv2.warpAffine(
        aligned_prediction,
        calibration_warp,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return np.clip(np.round(recap_grid * 255), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("work/chimera_composite_sim_dev256_audit.json"),
    )
    args = parser.parse_args()
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    photometry = json.loads(PHOTOMETRY.read_text(encoding="utf-8"))
    blur = json.loads(BLUR.read_text(encoding="utf-8"))
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        development = [
            row for row in csv.DictReader(stream) if row["split"] == "development"
        ]
    if len(development) != 120:
        raise RuntimeError("expected 120 development sources")
    args.output_root.mkdir(parents=True, exist_ok=True)
    image_hashes = []
    with args.manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("filename", "src", "label", "condition")
        )
        writer.writeheader()
        for source_row in development:
            for condition in ("recap_mac", "recap_monitor"):
                filename = f"{condition}/{source_row['src']}"
                destination = args.output_root / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(
                    render(source_row["src"], condition, geometry, photometry, blur),
                    mode="RGB",
                ).save(destination, format="PNG")
                with Image.open(destination) as check:
                    check.load()
                    if check.size != (256, 256) or check.mode != "RGB":
                        raise RuntimeError(
                            f"invalid materialized output: {destination}"
                        )
                writer.writerow(
                    {
                        "filename": filename,
                        "src": source_row["src"],
                        "label": source_row["label"],
                        "condition": condition,
                    }
                )
                image_hashes.append(
                    {"filename": filename, "sha256": file_hash(destination)}
                )
    audit = {
        "scope": "development-only composite published-output screen proxy",
        "root": str(args.output_root),
        "images": len(image_hashes),
        "manifest": str(args.manifest),
        "manifest_sha256": file_hash(args.manifest),
        "inputs_sha256": {
            str(path): file_hash(path) for path in (SPLIT, GEOMETRY, PHOTOMETRY, BLUR)
        },
        "image_sha256": image_hashes,
        "warning": "not a physical display/camera simulation; no reserved-source output generated",
    }
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: audit[key] for key in ("images", "manifest_sha256")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
