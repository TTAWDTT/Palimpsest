"""Conservative screen-lattice alias probe on verified Dragotti color-chart crops.

Conditional screen pitch comes from previous homography audit and a 1080-row
display assumption. This probe cannot identify real f-number or display frame.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from work.analyze_dragotti_probe import align, read_rgb


DATA = Path(r"E:\ai_image_origin_research\data\derived\dragotti_probe")
OUT = Path("work/dragotti_chart_alias_probe.json")
source = read_rgb(DATA / "original_D40_015.jpg")
SX = source.shape[1] / 1924.0
SY = source.shape[0] / 1280.0
# Centers selected inside visually flat ColorChecker fields in the 1924x1280
# inspection view; each is evaluated with a 64x64 output window.
centers_preview = (
    (195, 540), (350, 540), (495, 540), (625, 535), (755, 535), (865, 530),
    (215, 700), (365, 700), (510, 690), (650, 680), (775, 675), (885, 665),
    (240, 860), (385, 845), (530, 835), (670, 820), (800, 810), (915, 800),
)
AUDIT = json.loads(Path("outputs/03_过程模拟/Dragotti跨设备配对审计_2026-09-24.json").read_text(
    encoding="utf-8"))
records = AUDIT["samples"][0]["recaptured"]


def peak_ratio(patch: np.ndarray, expected: float) -> dict:
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    detrended = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(detrended * window)))**2
    fy = np.fft.fftshift(np.fft.fftfreq(gray.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(gray.shape[1]))
    xx, yy = np.meshgrid(fx, fy)
    radius = np.hypot(xx, yy)
    band = (radius >= expected - .025) & (radius <= expected + .025)
    background = (radius >= expected - .07) & (radius <= expected + .07) & ~band
    candidate = np.where(band, power, -np.inf)
    peak_y, peak_x = np.unravel_index(np.argmax(candidate), candidate.shape)
    return {"peak_to_annular_median": float(np.max(power[band]) /
                                             max(np.median(power[background]), 1e-12)),
            "band_mean_to_annular_mean": float(np.mean(power[band]) /
                                               max(np.mean(power[background]), 1e-12)),
            "peak_frequency_xy": [float(fx[peak_x]), float(fy[peak_y])]}


results = []
for camera in ("EOS600D", "D3200"):
    record = next(r for r in records if r["camera"] == camera)
    target = read_rgb(Path(record["file"]))
    homography, registration = align(source, target)
    warped = cv2.warpPerspective(source, homography, (target.shape[1], target.shape[0]),
                                 flags=cv2.INTER_CUBIC)
    k = record["conditional_projected_screen_pitch_if_fit_1080_rows"]
    alias = 1 - 1 / k
    patches = []
    for cx_small, cy_small in centers_preview:
        source_center = np.float32([[[cx_small * SX, cy_small * SY]]])
        cx, cy = cv2.perspectiveTransform(source_center, homography)[0, 0]
        x, y = int(round(cx)), int(round(cy))
        if min(x, y) < 32 or x + 32 > target.shape[1] or y + 32 > target.shape[0]:
            continue
        actual = target[y - 32:y + 32, x - 32:x + 32]
        digital = warped[y - 32:y + 32, x - 32:x + 32]
        real_score = peak_ratio(actual, alias)
        digital_score = peak_ratio(digital, alias)
        patches.append({"source_center_preview": [cx_small, cy_small],
                        "target_center": [x, y], "real": real_score,
                        "digital_warp_control": digital_score})
    results.append({"camera": camera, "registration_inliers": registration["ransac_inliers"],
                    "conditional_pitch": k, "conditional_alias_frequency": alias,
                    "n_patches": len(patches), "patches": patches,
                    "median_real_band_mean_ratio": float(np.median([
                        r["real"]["band_mean_to_annular_mean"] for r in patches])),
                    "median_digital_control_band_mean_ratio": float(np.median([
                        r["digital_warp_control"]["band_mean_to_annular_mean"] for r in patches])),
                    "peak_axis_concentration_real": float(abs(np.mean([
                        np.exp(2j * np.arctan2(r["real"]["peak_frequency_xy"][1],
                                                 r["real"]["peak_frequency_xy"][0]))
                        for r in patches]))),
                    "peak_axis_concentration_digital_control": float(abs(np.mean([
                        np.exp(2j * np.arctan2(r["digital_warp_control"]["peak_frequency_xy"][1],
                                                 r["digital_warp_control"]["peak_frequency_xy"][0]))
                        for r in patches])))})

report = {"status": "exploratory real-pixel frequency probe, not calibrated physical validation",
          "limitations": ["conditional pitch assumes 1080-row fit", "small chart windows",
                          "unknown display frame", "broad annulus contains scene/ISP spectra"],
          "results": results}
OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps([{k: r[k] for k in ("camera", "conditional_alias_frequency", "n_patches",
                                    "median_real_band_mean_ratio", "median_digital_control_band_mean_ratio",
                                    "peak_axis_concentration_real", "peak_axis_concentration_digital_control")}
                  for r in results], indent=2))
