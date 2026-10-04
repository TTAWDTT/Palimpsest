"""Small exploratory optical-focus intervention audit on FDNet author's examples.

Five released examples are already 256 px crops/resizes, not original camera
frames. The selected flat-region ROIs for 0003/0004 are diagnostic, not a
pre-registered population estimator or device calibration.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, WORK_DIR

import json

import cv2
import numpy as np
from PIL import Image


ROOT = DATA_ROOT / "raw/fdnet_real_screen_pairs"
OUT = WORK_DIR / "fdnet_focus_pair_process_probe.json"
ROIS = {
    "0003.png": {"moire": [82, 20, 165, 86], "content": [40, 110, 215, 218]},
    "0004.png": {"moire": [145, 24, 205, 100], "content": [43, 110, 215, 214]},
}


def load(condition: str, name: str) -> np.ndarray:
    return (
        np.asarray(Image.open(ROOT / condition / name).convert("RGB"), dtype=np.float32)
        / 255
    )


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def crop(image: np.ndarray, box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def fft_amplitude(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape
    residual = image - cv2.GaussianBlur(image, (0, 0), 12)
    window = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual * window)))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    return spectrum, fx, fy


def peak_info(focus: np.ndarray, defocus: np.ndarray) -> dict:
    spectrum, fx, fy = fft_amplitude(focus)
    other, _, _ = fft_amplitude(defocus)
    xx, yy = np.meshgrid(fx, fy)
    radius = np.hypot(xx, yy)
    mask = (radius >= 0.035) & (radius <= 0.22) & (xx >= 0)
    candidate = np.where(mask, spectrum, -np.inf)
    peak_y, peak_x = np.unravel_index(np.argmax(candidate), candidate.shape)
    target_fx, target_fy = float(fx[peak_x]), float(fy[peak_y])
    neighborhood = ((xx - target_fx) ** 2 + (yy - target_fy) ** 2) <= 0.018**2
    # Use local L2 amplitude to tolerate slight focus-breathing frequency shift.
    focus_power = float(np.sum(spectrum[neighborhood] ** 2))
    defocus_power = float(np.sum(other[neighborhood] ** 2))
    return {
        "frequency_fx_fy": [target_fx, target_fy],
        "cycles_per_pixel": float(np.hypot(target_fx, target_fy)),
        "period_pixels": float(1 / np.hypot(target_fx, target_fy)),
        "focus_power": focus_power,
        "defocus_power": defocus_power,
        "defocus_to_focus_power_ratio": defocus_power / focus_power,
        "neighborhood_radius_cycles_per_pixel": 0.018,
        "neighborhood_mask": neighborhood,
    }


def content_detail(image: np.ndarray, box: list[int]) -> float:
    patch = crop(image, box)
    # Smooth first to reduce native moire/noise and measure mid-scale details.
    smoothed = cv2.GaussianBlur(patch, (0, 0), 0.7)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3) / 8
    return float(np.mean(np.hypot(gx, gy)))


def main() -> None:
    names = [f"{index:04d}.png" for index in range(1, 6)]
    focused = {name: gray(load("moireresize", name)) for name in names}
    defocused = {name: gray(load("blurresize", name)) for name in names}
    pairing = []
    for name in names:
        a = cv2.GaussianBlur(focused[name][16:240, 16:240], (0, 0), 3).ravel()
        scores = {}
        for other in names:
            b = cv2.GaussianBlur(defocused[other][16:240, 16:240], (0, 0), 3).ravel()
            scores[other] = float(np.corrcoef(a, b)[0, 1])
        pairing.append(
            {
                "focused": name,
                "lowpass_pearson_by_defocused_name": scores,
                "best_match": max(scores, key=scores.get),
            }
        )
    interventions = []
    for name, boxes in ROIS.items():
        a, b = focused[name], defocused[name]
        peak = peak_info(crop(a, boxes["moire"]), crop(b, boxes["moire"]))
        neighborhood = peak.pop("neighborhood_mask")
        controls = []
        for sigma in np.arange(0, 10.01, 0.25):
            digital = cv2.GaussianBlur(a, (0, 0), float(sigma)) if sigma else a
            spectrum, _, _ = fft_amplitude(crop(digital, boxes["moire"]))
            ratio = float(np.sum(spectrum[neighborhood] ** 2) / peak["focus_power"])
            controls.append(
                {
                    "sigma_output_pixels": float(sigma),
                    "moire_peak_power_ratio": ratio,
                    "content_detail": content_detail(digital, boxes["content"]),
                }
            )
        best = min(
            controls,
            key=lambda row: abs(
                row["moire_peak_power_ratio"] - peak["defocus_to_focus_power_ratio"]
            ),
        )
        interventions.append(
            {
                "name": name,
                "rois_xyxy": boxes,
                "peak": peak,
                "focused_content_detail": content_detail(a, boxes["content"]),
                "real_defocused_content_detail": content_detail(b, boxes["content"]),
                "postblur_matched_peak": best,
                "controls": controls,
            }
        )
    report = {
        "scope": "5 author-released 256px real-screen focus/defocus examples",
        "pairing": pairing,
        "interventions": interventions,
        "qualification": "ROIs exploratory; target defocused image may differ by alignment/color/ISP, not isolated optical PSF; digital Gaussian control is post-capture only",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "pairing": pairing,
                "interventions": [
                    {
                        key: row[key]
                        for key in (
                            "name",
                            "peak",
                            "focused_content_detail",
                            "real_defocused_content_detail",
                            "postblur_matched_peak",
                        )
                    }
                    for row in interventions
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
