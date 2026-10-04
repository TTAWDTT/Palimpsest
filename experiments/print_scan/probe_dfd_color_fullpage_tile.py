"""Measure one color tile and blank-paper control in an 800-ppi DFD full scan.

The ROIs are fixed from a whole-page thumbnail, before inspecting their FFT.
The observed RGB peaks can be harmonics and are not identified RIP fundamentals.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import json

import cv2
import numpy as np
from PIL import Image

from experiments.print_scan.probe_dfd_color_sample_spectra import peaks


METADATA = WORK_DIR / "dfd_color_fullpage_metadata.json"
OUT = WORK_DIR / "dfd_color_fullpage_tile_probe.json"
FOLDER = "D2_HPM553"
ROIS = {
    "blue_color_tile": (3171, 3439, 3683, 3951),
    "blank_paper": (5000, 7000, 5512, 7512),
}


def main() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    record = next(row for row in metadata["records"] if row["folder"] == FOLDER)
    if record["encoded_dpi"] != [800.0, 800.0]:
        raise ValueError("physical peak conversion requires verified 800 ppi tag")
    observations = {}
    with Image.open(record["path"]) as image:
        for name, box in ROIS.items():
            crop = np.asarray(image.crop(box).convert("RGB"), dtype=np.float32) / 255
            if crop.shape != (512, 512, 3):
                raise ValueError(f"ROI outside page: {name}")
            native = peaks(crop, highpass_sigma_output_px=10)
            for channel in native:
                fx, fy = channel["peak_cycles_per_output_pixel_xy"]
                channel["observed_peak_radial_cycles_per_inch"] = float(
                    np.hypot(fx, fy) * 800
                )
            observations[name] = {
                "roi_xyxy": box,
                "channel_lowpass_spatial_std": cv2.GaussianBlur(crop, (0, 0), 32)
                .std((0, 1))
                .tolist(),
                "native": native,
                "area_downsample_4": peaks(
                    cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA),
                    highpass_sigma_output_px=2.5,
                ),
            }
    result = {
        "fullpage_sha256": record["sha256"],
        "member_name": record["archive_member"],
        "scanner_ppi_encoded": record["encoded_dpi"],
        "roi_selection": "visually fixed chart-blue center and blank right margin from reduced full-sheet preview",
        "warning": "strongest RGB peaks are composite scan observations; may be harmonics, aliases, or color-layer cross-talk",
        "observations": observations,
    }
    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: {
                    "native_highpass_std": [
                        round(ch["highpass_std"], 5) for ch in value["native"]
                    ],
                    "native_peak_cycles_per_inch": [
                        round(ch["observed_peak_radial_cycles_per_inch"], 1)
                        for ch in value["native"]
                    ],
                }
                for key, value in observations.items()
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
