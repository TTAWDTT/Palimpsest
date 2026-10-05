"""Fixed-parameter, non-calibrated print->paper->camera probe on three real pairs.

Only center patches are used; the public DIV2K-SCAN images have already been
perspective warped and aligned. This is a forward-chain smoke/falsification
probe, not a device-fidelity score or a parameter fit.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from palimpsest.simulation.color_print_scan import (
    ColorPrintScanParameters,
    simulate_color_print_surface,
)
from palimpsest.simulation.print_camera import (
    PrintCameraParameters,
    photograph_print_surface,
)
from palimpsest.simulation.publication import PublicationParameters, apply_publication


RAW = DATA_ROOT / "raw/div2k_scan"
SOURCE = DATA_ROOT / "derived/div2k_scan_original_probe"
OUTPUT = WORK_DIR / "div2k_print_camera_probe.json"
PATCH_SIDE = 256
PHOTO_WIDTH_INCHES = 7.5 / 2.54
DIGITAL_PPI = 2040 / PHOTO_WIDTH_INCHES


def _prepared_source(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        width, height = image.size
        # Author's stated center crop to 15:10, then place within published
        # 2040x1360 frame. Exact author's interpolation is not published.
        wanted_height = min(height, round(width * 2 / 3))
        wanted_width = min(width, round(height * 3 / 2))
        left = (width - wanted_width) // 2
        top = (height - wanted_height) // 2
        crop = image.crop((left, top, left + wanted_width, top + wanted_height))
        return np.asarray(
            crop.resize((2040, 1360), Image.Resampling.BICUBIC).convert("RGB")
        )


def _stats(image: np.ndarray) -> dict:
    rgb = np.asarray(image, dtype=np.float32) / 255
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return {
        "rgb_mean": rgb.mean((0, 1)).tolist(),
        "luma_mean": float(gray.mean()),
        "luma_std": float(gray.std()),
        "gradient_mean": float(
            np.mean(np.abs(np.diff(gray, axis=0)))
            + np.mean(np.abs(np.diff(gray, axis=1)))
        ),
    }


def main() -> None:
    print_params = ColorPrintScanParameters(
        digital_ppi=DIGITAL_PPI,
        render_ppi=1600,
        scan_ppi=800,
        screen_lpi=100,
        max_render_pixels=1_000_000,
    )
    camera_params = PrintCameraParameters(
        sensor_to_paper_inches=np.diag((1 / DIGITAL_PPI, 1 / DIGITAL_PPI, 1)),
        paper_ppi=print_params.render_ppi,
        ambient_irradiance_rgb=(1, 1, 1),
        optical_blur_sigma_sensor_pixels=0.5,
    )
    publish = PublicationParameters(encoding="png")
    rows = []
    with zipfile.ZipFile(RAW / "test_xr.zip") as archive:
        for image_id in ("0801", "0802", "0803"):
            clean = _prepared_source(SOURCE / f"{image_id}.png")
            with Image.open(io.BytesIO(archive.read(f"xr/{image_id}.png"))) as image:
                true_camera = np.asarray(image.convert("RGB"))
            y0, x0 = (1360 - PATCH_SIDE) // 2, (2040 - PATCH_SIDE) // 2
            source_patch = clean[y0 : y0 + PATCH_SIDE, x0 : x0 + PATCH_SIDE]
            real_patch = true_camera[y0 : y0 + PATCH_SIDE, x0 : x0 + PATCH_SIDE]
            surface = simulate_color_print_surface(source_patch, print_params)
            rendered = photograph_print_surface(
                surface.paper_reflectance_rgb, (PATCH_SIDE, PATCH_SIDE), camera_params
            )
            synthetic_patch = apply_publication(rendered.srgb, publish).decoded_rgb
            rows.append(
                {
                    "id": image_id,
                    "roi_xyxy": [x0, y0, x0 + PATCH_SIDE, y0 + PATCH_SIDE],
                    "paper_size_inches": surface.physical_size_inches,
                    "source": _stats(source_patch),
                    "real_publication": _stats(real_patch),
                    "simulated_publication": _stats(synthetic_patch),
                }
            )
    result = {
        "status": "uncalibrated fixed-parameter center-patch probe",
        "photo_width_inches_from_paper": PHOTO_WIDTH_INCHES,
        "digital_ppi_inferred_for_author_prepared_frame": DIGITAL_PPI,
        "print_parameters": print_params.__dict__,
        "camera_parameters": {
            **camera_params.__dict__,
            "sensor_to_paper_inches": camera_params.sensor_to_paper_inches.tolist(),
        },
        "rows": rows,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            row["id"],
            "source",
            round(row["source"]["luma_mean"], 3),
            "real",
            round(row["real_publication"]["luma_mean"], 3),
            "sim",
            round(row["simulated_publication"]["luma_mean"], 3),
        )


if __name__ == "__main__":
    main()
