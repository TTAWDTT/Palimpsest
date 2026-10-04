"""Pre-fixed four-color ROI spectral peaks across D5/D6 same nominal print setup.

This tests whether output lattice candidates persist across printer IDs. It is
not a fitted forward simulator or CMYK color-separation recovery.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import json

import cv2
import numpy as np
from PIL import Image

from experiments.print_scan.probe_dfd_color_matched_printers import top_spectral_peaks


METADATA = WORK_DIR / "dfd_color_fullpage_metadata.json"
OUT = WORK_DIR / "dfd_color_multitile_transfer.json"
FOLDERS = ("D5_HPCLJ5550", "D6_HPCLJ5550")
ROIS = {
    "red": (1020, 3439, 1532, 3951),
    "green": (2070, 3439, 2582, 3951),
    "blue": (3171, 3439, 3683, 3951),
    "yellow": (3171, 4470, 3683, 4982),
}
MATCH_RADIUS_FFT_BINS = 2.0


def peak_distance_bins(a: dict, b: dict) -> float:
    vector_a = np.asarray(a["cycles_per_pixel_xy"], dtype=float)
    vector_b = np.asarray(b["cycles_per_pixel_xy"], dtype=float)
    return float(np.linalg.norm(vector_a - vector_b) * 512)


def main() -> None:
    records = json.loads(METADATA.read_text(encoding="utf-8"))["records"]
    scans = {}
    for folder in FOLDERS:
        record = next(row for row in records if row["folder"] == folder)
        if (
            record["encoded_dpi"] != [800, 800]
            or "_W1_600_800_P1_" not in record["archive_member"]
        ):
            raise ValueError("matched nominal setting or scan scale changed")
        observations = {}
        with Image.open(record["path"]) as image:
            for color, box in ROIS.items():
                patch = (
                    np.asarray(image.crop(box).convert("RGB"), dtype=np.float32) / 255
                )
                if patch.shape != (512, 512, 3):
                    raise ValueError("fixed chart ROI outside scan")
                lowpass_std = cv2.GaussianBlur(patch, (0, 0), 32).std((0, 1))
                observations[color] = {
                    "lowpass_spatial_std_rgb": lowpass_std.tolist(),
                    "top_three": {
                        channel: top_spectral_peaks(patch, index, limit=3)
                        for index, channel in enumerate("RGB")
                    },
                }
        scans[folder] = {
            "member_name": record["archive_member"],
            "sha256": record["sha256"],
            "color_rois": observations,
        }
    comparisons = []
    for color in ROIS:
        for channel in "RGB":
            one = scans[FOLDERS[0]]["color_rois"][color]["top_three"][channel]
            other = scans[FOLDERS[1]]["color_rois"][color]["top_three"][channel]
            first_distance = peak_distance_bins(one[0], other[0])
            nearest = min(peak_distance_bins(one[0], candidate) for candidate in other)
            comparisons.append(
                {
                    "color": color,
                    "channel": channel,
                    "d5_top1_to_d6_top1_distance_fft_bins": first_distance,
                    "d5_top1_to_d6_top3_nearest_fft_bins": nearest,
                    "top1_match_within_2_bins": first_distance <= MATCH_RADIUS_FFT_BINS,
                    "top3_contains_d5_top1_within_2_bins": nearest
                    <= MATCH_RADIUS_FFT_BINS,
                }
            )
    report = {
        "selection": "four chart patches fixed from whole-page preview before spectral comparison; both sheets encode 800 ppi",
        "matching_rule": "distance <=2 FFT bins in 512x512 output; top peaks ranked by central-bin power",
        "limitations": "one pair of devices, unknown remaining process settings; RGB peak vectors can be mixed/harmonic",
        "rois": ROIS,
        "scans": scans,
        "comparisons": comparisons,
        "summary": {
            "channels": len(comparisons),
            "top1_matches": sum(r["top1_match_within_2_bins"] for r in comparisons),
            "top3_contains_d5_top1": sum(
                r["top3_contains_d5_top1_within_2_bins"] for r in comparisons
            ),
        },
    }
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"summary": report["summary"], "comparisons": comparisons}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
