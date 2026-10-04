"""Exploratory 2D spectra of one selected DFD color scan per printer folder.

These are observed RGB scan frequencies. Without encoded DPI or a chart source,
they are not printer screen frequencies or CMYK separation estimates.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


INVENTORY = WORK_DIR / "dfd_color_inventory.json"
OUT = WORK_DIR / "dfd_color_selected_spectra.json"


def peaks(rgb: np.ndarray, *, highpass_sigma_output_px: float) -> list[dict]:
    size = rgb.shape[0]
    window = np.outer(np.hanning(size), np.hanning(size))
    y, x = np.mgrid[-size // 2 : size // 2, -size // 2 : size // 2]
    radial = np.hypot(x, y) / size
    # One side removes the redundant complex-conjugate peak.
    half_plane = (y < 0) | ((y == 0) & (x > 0))
    support = (radial >= 0.05) & (radial <= 0.45) & half_plane
    result = []
    for channel in range(3):
        image = rgb[:, :, channel]
        residual = image - cv2.GaussianBlur(image, (0, 0), highpass_sigma_output_px)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual * window))) ** 2
        candidate = np.where(support, spectrum, 0)
        row, col = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        surrounding = spectrum[
            max(0, row - 1) : min(size, row + 2), max(0, col - 1) : min(size, col + 2)
        ]
        denominator = max(float(spectrum[support].sum()), 1e-20)
        result.append(
            {
                "channel": "RGB"[channel],
                "peak_cycles_per_output_pixel_xy": [
                    (col - size // 2) / size,
                    (row - size // 2) / size,
                ],
                "peak_neighborhood_power_fraction_of_half_annulus": float(
                    surrounding.sum() / denominator
                ),
                "mean": float(image.mean()),
                "std": float(image.std()),
                "highpass_std": float(residual.std()),
            }
        )
    return result


def main() -> None:
    audit = json.loads(INVENTORY.read_text(encoding="utf-8"))
    results = []
    for selected in audit["selected_samples"]:
        path = Path(selected["path"])
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        h, w = rgb.shape[:2]
        if min(h, w) < 512:
            results.append(
                {
                    "member_name": selected["member_name"],
                    "status": "skip: smaller than fixed 512 pixel window",
                }
            )
            continue
        top, left = (h - 512) // 2, (w - 512) // 2
        crop = rgb[top : top + 512, left : left + 512]
        low_frequency_spatial_std = cv2.GaussianBlur(crop, (0, 0), 32).std((0, 1))
        if float(low_frequency_spatial_std.max()) >= 0.03:
            results.append(
                {
                    "member_name": selected["member_name"],
                    "status": "skip: nonuniform central region",
                    "max_channel_lowpass_spatial_std": float(
                        low_frequency_spatial_std.max()
                    ),
                }
            )
            continue
        results.append(
            {
                "member_name": selected["member_name"],
                "sample_sha256": selected["sha256"],
                "encoded_dpi": selected["encoded_dpi"],
                "crop_xyxy": [left, top, left + 512, top + 512],
                "max_channel_lowpass_spatial_std": float(
                    low_frequency_spatial_std.max()
                ),
                "native_512": peaks(crop, highpass_sigma_output_px=10),
                "area_downsample_2": peaks(
                    cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA),
                    highpass_sigma_output_px=5,
                ),
                "area_downsample_4": peaks(
                    cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA),
                    highpass_sigma_output_px=2.5,
                ),
            }
        )
    usable = [row for row in results if "native_512" in row]
    highpass_ratio_4 = [
        row["area_downsample_4"][channel]["highpass_std"]
        / max(row["native_512"][channel]["highpass_std"], 1e-12)
        for row in usable
        for channel in range(3)
    ]
    peak_fraction_native = [
        row["native_512"][channel]["peak_neighborhood_power_fraction_of_half_annulus"]
        for row in usable
        for channel in range(3)
    ]
    report = {
        "scope": "one archive-order selected scan per printer folder; observational RGB output spectra, not source-to-output calibration",
        "uniformity_rule": "central 512 crop; max RGB channel spatial std after Gaussian sigma 32 pixels <0.03; threshold chosen as exploratory chart/photo guard",
        "window": "central 512x512; high-pass Gaussian sigma 10 native pixels, scaled to 5/2.5 after area downsample; Hann FFT; half-annulus .05-.45 cycles/output pixel",
        "summary": {
            "uniform_crops_analyzed": len(usable),
            "median_native_rgb_peak_fraction": float(np.median(peak_fraction_native)),
            "median_area4_to_native_highpass_std_ratio": float(
                np.median(highpass_ratio_4)
            ),
        },
        "results": results,
    }
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scans": len(results),
                "analyzed": sum("native_512" in r for r in results),
                "native_peak_fractions": {
                    r["member_name"]: [
                        round(v["peak_neighborhood_power_fraction_of_half_annulus"], 4)
                        for v in r["native_512"]
                    ]
                    for r in results
                    if "native_512" in r
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
