"""Fixed continuous-tone print hypothesis on the three verified DIV2K-SCAN pairs.

This is a mechanism comparison with the earlier ordered-CMYK probe, not a
parameter fit. The camera and publication parameters are identical.
"""

from __future__ import annotations

from experiments.paths import WORK_DIR

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from origin_simulation.photo_paper import (
    PhotoPaperParameters,
    simulate_photo_paper_surface,
)
from origin_simulation.print_camera import (
    PrintCameraParameters,
    photograph_print_surface,
)
from origin_simulation.publication import PublicationParameters, apply_publication
from experiments.print_scan.probe_print_camera_div2k import (
    RAW,
    SOURCE,
    PATCH_SIDE,
    DIGITAL_PPI,
    PHOTO_WIDTH_INCHES,
    _prepared_source,
    _stats,
)

OUTPUT = WORK_DIR / "div2k_photo_paper_probe.json"


def main() -> None:
    paper_params = PhotoPaperParameters(
        digital_ppi=DIGITAL_PPI,
        render_ppi=1600,
        exposure_spot_sigma_um=12,
        dye_spread_sigma_um=8,
        grain_correlation_um=3,
        grain_density_std=0.02,
        max_render_pixels=1_000_000,
    )
    camera_params = PrintCameraParameters(
        sensor_to_paper_inches=np.diag((1 / DIGITAL_PPI, 1 / DIGITAL_PPI, 1)),
        paper_ppi=paper_params.render_ppi,
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
            surface = simulate_photo_paper_surface(source_patch, paper_params)
            rendered = photograph_print_surface(
                surface.paper_reflectance_rgb, (PATCH_SIDE, PATCH_SIDE), camera_params
            )
            synthetic_patch = apply_publication(rendered.srgb, publish).decoded_rgb
            rows.append(
                {
                    "id": image_id,
                    "roi_xyxy": [x0, y0, x0 + PATCH_SIDE, y0 + PATCH_SIDE],
                    "source": _stats(source_patch),
                    "real_publication": _stats(real_patch),
                    "simulated_publication": _stats(synthetic_patch),
                }
            )
    result = {
        "status": "uncalibrated continuous-tone physical hypothesis probe",
        "physical_route": "effective laser spot -> developed three-dye density/grain -> paper -> camera -> PNG",
        "photo_width_inches_from_paper": PHOTO_WIDTH_INCHES,
        "digital_ppi_inferred_for_author_prepared_frame": DIGITAL_PPI,
        "paper_parameters": paper_params.__dict__,
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
            "real",
            round(row["real_publication"]["luma_mean"], 3),
            "sim",
            round(row["simulated_publication"]["luma_mean"], 3),
            "grad real/sim",
            round(row["real_publication"]["gradient_mean"], 3),
            round(row["simulated_publication"]["gradient_mean"], 3),
        )


if __name__ == "__main__":
    main()
