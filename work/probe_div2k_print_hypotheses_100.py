"""Fixed, uncalibrated print->camera hypotheses on audited DIV2K-SCAN pairs.

This extends the identical three-pair probe to 100 content-audited sources.
Only center 256px patches and coarse luma/gradient descriptors are compared;
no registered-pixel error or printer technology identification is claimed.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from origin_simulation.color_print_scan import ColorPrintScanParameters, simulate_color_print_surface
from origin_simulation.photo_paper import PhotoPaperParameters, simulate_photo_paper_surface
from origin_simulation.print_camera import PrintCameraParameters, photograph_print_surface
from origin_simulation.publication import PublicationParameters, apply_publication
from work.audit_div2k_scan_test import RAW, _source_3_2
from work.probe_print_camera_div2k import DIGITAL_PPI, PATCH_SIDE, PHOTO_WIDTH_INCHES, _stats


AUDIT = Path(__file__).with_name("div2k_scan_100_pair_audit.json")
OUT = Path(__file__).with_name("div2k_print_hypotheses_100.json")


def simulate(source_patch: np.ndarray, route: str,
             parameters, camera: PrintCameraParameters,
             publish: PublicationParameters) -> np.ndarray:
    if route == "ordered_cmyk":
        surface = simulate_color_print_surface(source_patch, parameters)
    elif route == "continuous_tone_proxy":
        surface = simulate_photo_paper_surface(source_patch, parameters)
    else:
        raise ValueError(route)
    rendered = photograph_print_surface(surface.paper_reflectance_rgb,
                                        (PATCH_SIDE, PATCH_SIDE), camera)
    return apply_publication(rendered.srgb, publish).decoded_rgb


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["all_members_crc_passed"] or len(audit["rows"]) != 100:
        raise ValueError("100-pair source/capture audit not completed")
    ordered = ColorPrintScanParameters(digital_ppi=DIGITAL_PPI, render_ppi=1600,
                                       scan_ppi=800, screen_lpi=100,
                                       max_render_pixels=1_000_000)
    continuous = PhotoPaperParameters(digital_ppi=DIGITAL_PPI, render_ppi=1600,
                                      exposure_spot_sigma_um=12,
                                      dye_spread_sigma_um=8,
                                      grain_correlation_um=3,
                                      grain_density_std=.02,
                                      max_render_pixels=1_000_000)
    camera = PrintCameraParameters(
        sensor_to_paper_inches=np.diag((1/DIGITAL_PPI, 1/DIGITAL_PPI, 1)),
        paper_ppi=1600, ambient_irradiance_rgb=(1, 1, 1),
        optical_blur_sigma_sensor_pixels=.5)
    publish = PublicationParameters(encoding="png")
    rows = []
    with (zipfile.ZipFile(RAW / "DIV2K_valid_HR.zip") as sources,
          zipfile.ZipFile(RAW / "test_xr.zip") as captures):
        for index, audit_row in enumerate(audit["rows"], 1):
            image_id = audit_row["id"]
            with Image.open(io.BytesIO(sources.read(f"DIV2K_valid_HR/{image_id}.png"))) as image:
                source = _source_3_2(image, (2040, 1360))
            with Image.open(io.BytesIO(captures.read(f"xr/{image_id}.png"))) as image:
                real = np.asarray(image.convert("RGB"))
            y0, x0 = (1360 - PATCH_SIDE)//2, (2040 - PATCH_SIDE)//2
            source_patch = source[y0:y0+PATCH_SIDE, x0:x0+PATCH_SIDE]
            real_patch = real[y0:y0+PATCH_SIDE, x0:x0+PATCH_SIDE]
            cmyk = simulate(source_patch, "ordered_cmyk", ordered, camera, publish)
            photo = simulate(source_patch, "continuous_tone_proxy", continuous, camera, publish)
            rows.append({"id": image_id, "content_evidence": audit_row["content_evidence"],
                         "source": _stats(source_patch), "real_publication": _stats(real_patch),
                         "ordered_cmyk": _stats(cmyk),
                         "continuous_tone_proxy": _stats(photo)})
            if index % 10 == 0:
                print(f"fixed print hypotheses {index}/100", flush=True)
    summaries = {}
    for route in ("ordered_cmyk", "continuous_tone_proxy"):
        real_luma = np.array([row["real_publication"]["luma_mean"] for row in rows])
        sim_luma = np.array([row[route]["luma_mean"] for row in rows])
        real_grad = np.array([row["real_publication"]["gradient_mean"] for row in rows])
        sim_grad = np.array([row[route]["gradient_mean"] for row in rows])
        summaries[route] = {"median_luma_real": float(np.median(real_luma)),
                            "median_luma_sim": float(np.median(sim_luma)),
                            "median_luma_bias_sim_minus_real": float(np.median(sim_luma-real_luma)),
                            "sim_brighter_fraction": float(np.mean(sim_luma > real_luma)),
                            "median_gradient_real": float(np.median(real_grad)),
                            "median_gradient_sim": float(np.median(sim_grad)),
                            "median_gradient_ratio_sim_over_real":
                                float(np.median(sim_grad / np.maximum(real_grad, 1e-6)))}
    result = {"status": "fixed uncalibrated hypotheses, no technology identification",
              "source_archive_sha256": audit["source_archive_sha256"],
              "capture_archive_sha256": audit["capture_archive_sha256"],
              "source_ids": [row["id"] for row in rows],
              "photo_width_inches": PHOTO_WIDTH_INCHES,
              "prepared_digital_ppi": DIGITAL_PPI,
              "patch": {"side": PATCH_SIDE, "position": "center in aligned published frame"},
              "ordered_cmyk_parameters": ordered.__dict__,
              "continuous_tone_parameters": continuous.__dict__,
              "camera_parameters": {**camera.__dict__,
                                    "sensor_to_paper_inches": camera.sensor_to_paper_inches.tolist()},
              "summary": summaries, "rows": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
