"""Test the existing screen forward renderer against two known-source real RAW frames.

All optical/display settings are predeclared virtual hypotheses. The fit only
maps each sensor CFA plane's simulated irradiance to 10-bit counts using the
automobile capture; the airplane capture is a cross-content check.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from origin_simulation.screen_capture import ScreenCaptureParameters, render_screen_capture
from origin_simulation.screen_pipeline import DisplayRasterParameters, rasterize_display_source
from work.audit_raw2event_probe import ROOT, WIDTH, HEIGHT, detect_tag, extract_frame


PREFIXES = (
    "10000_automobile_5_1087_20251224_105416",
    "1000_airplane_1_9934_20251222_161953",
)
AUDITS = (
    Path("work/raw2event_probe_pixel_audit.json"),
    Path("work/raw2event_probe_airplane_pixel_audit.json"),
)
SOURCE_DIR = Path("work/raw2event_cifar_matches")
RGB_CORNERS = np.asarray([[237, 180], [409, 195], [397, 370], [219, 353]], dtype=np.float32)
DISPLAY_SIDE = 192  # explicit assumed display lattice; actual screen raster unpublished
BLUR_SIGMA_SENSOR_PIXELS = 0.8  # virtual optical PSF, not device-measured
INNER_FRACTION = 0.06
OUT = Path("work/raw2event_source_to_raw_probe.json")


def prepare(prefix: str, audit_path: Path | None = None, source_rgb: np.ndarray | None = None) -> dict:
    raw = extract_frame(ROOT / "frames_raw" / f"{prefix}.mkv", 0, "gray16le", 1, "<u2")
    if audit_path is None:
        rgb = extract_frame(ROOT / "frames_rgb" / f"{prefix}.mkv", 0, "rgb24", 3, "u1")
        raw_view = np.minimum(raw.astype(np.float32) / 1023 * 255, 255).astype(np.uint8)
        raw_tag, raw_id = detect_tag(raw_view)
        rgb_tag, rgb_id = detect_tag(rgb)
        if raw_id != rgb_id or raw_id != 0:
            raise RuntimeError(f"unexpected AprilTag association for {prefix}")
        h_raw_to_rgb = cv2.getPerspectiveTransform(raw_tag, rgb_tag)
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        h_raw_to_rgb = np.asarray(audit["samples"]["0"]["tag"]["raw_to_rgb_tag_homography"])
    raw_corners = cv2.perspectiveTransform(RGB_CORNERS[None], np.linalg.inv(h_raw_to_rgb))[0]
    left, top = np.floor(raw_corners.min(axis=0) - 8).astype(int)
    right, bottom = np.ceil(raw_corners.max(axis=0) + 8).astype(int)
    left, top = max(0, left), max(0, top)
    # The renderer starts its RGGB hypothesis at local (0,0); preserve the
    # global sensor parity when extracting a RAW ROI.
    left -= left % 2
    top -= top % 2
    right, bottom = min(WIDTH, right), min(HEIGHT, bottom)
    roi = (left, top, right, bottom)
    local = raw_corners - np.asarray([left, top], dtype=np.float32)
    display_corners = np.asarray([[0, 0], [DISPLAY_SIDE, 0], [DISPLAY_SIDE, DISPLAY_SIDE], [0, DISPLAY_SIDE]],
                                 dtype=np.float32)
    sensor_to_display = cv2.getPerspectiveTransform(local, display_corners)
    yy, xx = np.indices((bottom - top, right - left), dtype=np.float32)
    projected = cv2.perspectiveTransform(np.stack((xx + 0.5, yy + 0.5), axis=-1).reshape(1, -1, 2),
                                         sensor_to_display).reshape(bottom - top, right - left, 2)
    lower, upper = DISPLAY_SIDE * INNER_FRACTION, DISPLAY_SIDE * (1 - INNER_FRACTION)
    mask = ((projected[..., 0] >= lower) & (projected[..., 0] < upper) &
            (projected[..., 1] >= lower) & (projected[..., 1] < upper))
    source = (np.asarray(Image.open(SOURCE_DIR / f"{prefix}_source32.png").convert("RGB"))
              if source_rgb is None else np.asarray(source_rgb, dtype=np.uint8))
    drive = rasterize_display_source(source, DisplayRasterParameters(
        raster_size=(DISPLAY_SIDE, DISPLAY_SIDE), resampling="lanczos", resample_space="encoded_srgb"))
    return {"prefix": prefix, "roi": roi, "raw_corners": raw_corners, "H": sensor_to_display,
            "mask": mask, "actual": raw[top:bottom, left:right].astype(np.float32), "drive": drive}


def render(prepared: dict, emitter_layout: str) -> tuple[np.ndarray, float]:
    shape = prepared["actual"].shape
    params = ScreenCaptureParameters(
        sensor_to_display=prepared["H"], fill_fraction=0.85, emitter_layout=emitter_layout,
        display_gamma=2.2, optical_blur_sigma_sensor_pixels=BLUR_SIGMA_SENSOR_PIXELS,
        exposure_electrons_per_unit=None, read_noise_electrons=0.0)
    start = time.perf_counter()
    result = render_screen_capture(prepared["drive"], shape, params,
                                   spatial_method="fine", tile_size_sensor_pixels=96)
    return result.noiseless_mosaic, time.perf_counter() - start


def direct_sample_control(prepared: dict) -> tuple[np.ndarray, float]:
    """Digital interpolation plus Gaussian blur, with no emitter geometry."""
    start = time.perf_counter()
    height, width = prepared["actual"].shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    xy = np.stack((xx + 0.5, yy + 0.5), axis=-1).reshape(1, -1, 2)
    uv = cv2.perspectiveTransform(xy, prepared["H"]).reshape(height, width, 2)
    radiance = prepared["drive"] ** 2.2
    mapped = cv2.remap(radiance, uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mapped = cv2.GaussianBlur(mapped, (0, 0), BLUR_SIGMA_SENSOR_PIXELS)
    y, x = np.indices((height, width))
    channel = np.where((y % 2 == 0) & (x % 2 == 0), 0,
                       np.where((y % 2 == 1) & (x % 2 == 1), 2, 1))
    mosaic = np.take_along_axis(mapped, channel[..., None], axis=-1)[..., 0]
    return mosaic, time.perf_counter() - start


def cfa_design(simulated: np.ndarray) -> np.ndarray:
    y, x = np.indices(simulated.shape)
    r = (y % 2 == 0) & (x % 2 == 0)
    b = (y % 2 == 1) & (x % 2 == 1)
    g = ~(r | b)
    return np.stack((np.ones_like(simulated), simulated * r, simulated * g, simulated * b), axis=-1)


def fit_counts(simulated: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    design = cfa_design(simulated)[mask]
    return np.linalg.lstsq(design, actual[mask], rcond=None)[0]


def evaluate(simulated: np.ndarray, actual: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> dict:
    predicted = cfa_design(simulated) @ weights
    residual = predicted[mask] - actual[mask]
    return {"n": int(mask.sum()), "mae_counts": float(np.abs(residual).mean()),
            "rmse_counts": float(np.sqrt(np.mean(residual**2))),
            "pearson": float(np.corrcoef(predicted[mask], actual[mask])[0, 1]),
            "actual_mean": float(actual[mask].mean()), "predicted_mean": float(predicted[mask].mean())}


def main() -> None:
    calibration, heldout = [prepare(prefix, audit) for prefix, audit in zip(PREFIXES, AUDITS)]
    result = {"scope": "two known CIFAR sources and first-frame raw mosaics; one calibration, one heldout",
              "fixed_assumptions": {"display_side_pixels": DISPLAY_SIDE,
                                    "display_resampling": "Lanczos in encoded sRGB",
                                    "display_gamma": 2.2, "fill_fraction": 0.85,
                                    "optical_blur_sigma_sensor_pixels": BLUR_SIGMA_SENSOR_PIXELS,
                                    "raw_cfa_phase": "RGGB hypothesis, not verified",
                                    "geometry": "manual RGB content corners + per-recording AprilTag RAW/RGB homography",
                                    "roi_inner_fraction": INNER_FRACTION},
              "conditions": {}}
    for layout in ("vertical_rgb", "co_spatial_rgb_control", "simple_rgb_sample_control"):
        print(f"rendering {layout} automobile", flush=True)
        cal_sim, cal_seconds = (direct_sample_control(calibration) if layout == "simple_rgb_sample_control"
                                else render(calibration, layout))
        print(f"rendering {layout} airplane", flush=True)
        test_sim, test_seconds = (direct_sample_control(heldout) if layout == "simple_rgb_sample_control"
                                  else render(heldout, layout))
        weights = fit_counts(cal_sim, calibration["actual"], calibration["mask"])
        result["conditions"][layout] = {
            "counts_fit": {"intercept": float(weights[0]), "r_gain": float(weights[1]),
                           "g_gain": float(weights[2]), "b_gain": float(weights[3])},
            "calibration": evaluate(cal_sim, calibration["actual"], calibration["mask"], weights),
            "heldout": evaluate(test_sim, heldout["actual"], heldout["mask"], weights),
            "render_seconds": {"automobile": cal_seconds, "airplane": test_seconds},
        }
    result["rois"] = {item["prefix"]: {"xyxy": [int(value) for value in item["roi"]],
                                        "raw_content_corners": item["raw_corners"].tolist(),
                                        "interior_n": int(item["mask"].sum())}
                      for item in (calibration, heldout)}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
