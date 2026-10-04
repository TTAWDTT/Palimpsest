"""Same-named color chart/driver setting across two HP CLJ5550 printer IDs.

This is a two-scan observational check of effective halftone vectors. Identical
nominal filenames do not certify identical RIP, toner, paper, or scanner optics.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import json

import cv2
import numpy as np
from PIL import Image

from experiments.print_scan.probe_dfd_color_sample_spectra import peaks


METADATA = WORK_DIR / "dfd_color_fullpage_metadata.json"
OUT = WORK_DIR / "dfd_color_matched_printers.json"
FOLDERS = ("D5_HPCLJ5550", "D6_HPCLJ5550")
BLUE = (3171, 3439, 3683, 3951)
PAPER = (5000, 7000, 5512, 7512)


def top_spectral_peaks(rgb: np.ndarray, channel: int, limit: int = 5) -> list[dict]:
    size = rgb.shape[0]
    residual = rgb[:, :, channel] - cv2.GaussianBlur(rgb[:, :, channel], (0, 0), 10)
    window = np.outer(np.hanning(size), np.hanning(size))
    power = np.abs(np.fft.fftshift(np.fft.fft2(residual * window))) ** 2
    y, x = np.mgrid[-size // 2 : size // 2, -size // 2 : size // 2]
    band = (np.hypot(x, y) / size >= 0.05) & (np.hypot(x, y) / size <= 0.45)
    band &= (y < 0) | ((y == 0) & (x > 0))
    candidates = np.where(band, power, 0).astype(np.float32)
    local = candidates == cv2.dilate(candidates, np.ones((9, 9), np.uint8))
    indices = np.flatnonzero(local & band)
    indices = indices[np.argsort(candidates.flat[indices])[::-1]]
    found = []
    for index in indices:
        row, col = np.unravel_index(int(index), power.shape)
        if any((row - item[0]) ** 2 + (col - item[1]) ** 2 < 8**2 for item in found):
            continue
        found.append((row, col))
        if len(found) == limit:
            break
    total = max(float(power[band].sum()), 1e-20)
    return [
        {
            "cycles_per_pixel_xy": [(col - size // 2) / size, (row - size // 2) / size],
            "central_bin_power_fraction": float(power[row, col] / total),
            "three_by_three_power_fraction": float(
                power[row - 1 : row + 2, col - 1 : col + 2].sum() / total
            ),
        }
        for row, col in found
    ]


def main() -> None:
    records = json.loads(METADATA.read_text(encoding="utf-8"))["records"]
    scans = []
    for folder in FOLDERS:
        record = next(row for row in records if row["folder"] == folder)
        if record["encoded_dpi"] != [800, 800]:
            raise ValueError(f"physical scale unverified for {folder}")
        if "_W1_600_800_P1_" not in record["archive_member"]:
            raise ValueError(f"nominal setting differs for {folder}")
        observations = {}
        with Image.open(record["path"]) as image:
            for label, box in (("blue", BLUE), ("paper", PAPER)):
                patch = (
                    np.asarray(image.crop(box).convert("RGB"), dtype=np.float32) / 255
                )
                if patch.shape != (512, 512, 3):
                    raise ValueError("fixed crop unavailable")
                observations[label] = {
                    "dominant": peaks(patch, highpass_sigma_output_px=10),
                    "top_five_per_rgb": {
                        channel: top_spectral_peaks(patch, index)
                        for index, channel in enumerate("RGB")
                    },
                }
        scans.append(
            {
                "printer_id": folder,
                "member_name": record["archive_member"],
                "sha256": record["sha256"],
                "encoded_dpi": record["encoded_dpi"],
                "observations": observations,
            }
        )
    deltas = []
    for index, channel in enumerate("RGB"):
        a = np.asarray(
            scans[0]["observations"]["blue"]["dominant"][index][
                "peak_cycles_per_output_pixel_xy"
            ],
            dtype=np.float64,
        )
        b = np.asarray(
            scans[1]["observations"]["blue"]["dominant"][index][
                "peak_cycles_per_output_pixel_xy"
            ],
            dtype=np.float64,
        )
        deltas.append(
            {
                "channel": channel,
                "peak_vector_d5_cycles_per_pixel": a.tolist(),
                "peak_vector_d6_cycles_per_pixel": b.tolist(),
                "vector_distance_cycles_per_pixel": float(np.linalg.norm(a - b)),
                "radial_frequency_d5_cycles_per_inch": float(np.linalg.norm(a) * 800),
                "radial_frequency_d6_cycles_per_inch": float(np.linalg.norm(b) * 800),
            }
        )
    first_blue = scans[0]["observations"]["blue"]["dominant"]
    red_vector = np.asarray(
        first_blue[0]["peak_cycles_per_output_pixel_xy"], dtype=float
    )
    blue_vector = np.asarray(
        first_blue[2]["peak_cycles_per_output_pixel_xy"], dtype=float
    )
    basis_angle = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(red_vector, blue_vector)
                    / (np.linalg.norm(red_vector) * np.linalg.norm(blue_vector)),
                    -1,
                    1,
                )
            )
        )
    )
    report = {
        "selection": "two extracted whole pages of same model and same W1_600_800_P1 filename fields; fixed visually chosen chart-blue and blank-paper ROIs",
        "limitations": "one matched nominal setting, 2 device IDs; unknown physical paper/toner/RIP identity; output RGB peaks may be harmonic or cross-layer mixtures",
        "rois": {"blue": BLUE, "paper": PAPER},
        "scans": scans,
        "blue_peak_comparison": deltas,
        "angle_between_d5_blue_tile_rgb_red_blue_dominant_vectors_degrees": basis_angle,
    }
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "blue_peak_comparison": deltas,
                "blue_peak_fractions": {
                    scan["printer_id"]: [
                        round(
                            row["peak_neighborhood_power_fraction_of_half_annulus"], 4
                        )
                        for row in scan["observations"]["blue"]["dominant"]
                    ]
                    for scan in scans
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
