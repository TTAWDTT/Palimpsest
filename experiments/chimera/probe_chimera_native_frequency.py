"""Search for repeatable native-pixel spectral peaks in true screen recaptures.

Calibration identifies a candidate frequency. Development tests that *fixed*
frequency against a geometry/resize-only digital control. A peak is not proof
of display pixels: CFA, ISP and publication sampling remain alternative causes.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("E:/ai_image_origin_research/data/derived/chimera_paired")
SPLIT = Path(
    "E:/ai_image_origin_research/data/manifests/chimera_simulation_source_split.csv"
)
GEOMETRY = Path("work/chimera_fixed_publication_geometry.json")
OUTPUT = Path("work/chimera_native_frequency_probe.json")
PATCH = 160


def median_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot decode: {path}")
    return image.astype(np.float32) / 255


def flattest_source_center(source: np.ndarray) -> tuple[float, float]:
    gx = cv2.Sobel(source, cv2.CV_32F, 1, 0) / 8
    gy = cv2.Sobel(source, cv2.CV_32F, 0, 1) / 8
    gradient = np.hypot(gx, gy)
    candidates = [
        (float(gradient[y : y + 64, x : x + 64].mean()), y, x)
        for y in range(16, 177, 16)
        for x in range(16, 177, 16)
    ]
    _, y, x = min(candidates)
    return x + 32.0, y + 32.0


def native_patches(
    source_id: str, condition: str, warp: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    source_bgr = cv2.imread(str(ROOT / "stylegan2_orig" / source_id), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise RuntimeError(f"cannot decode source: {source_id}")
    source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    native = load_gray(ROOT / condition / source_id)
    expected = (1026, 1026) if condition == "recap_mac" else (765, 765)
    if native.shape != expected:
        raise RuntimeError(f"native size changed: {source_id} {condition}")
    source_x, source_y = flattest_source_center(source)
    center_x = (
        (warp[0, 0] * source_x + warp[0, 1] * source_y + warp[0, 2])
        * native.shape[1]
        / 256
    )
    center_y = (
        (warp[1, 0] * source_x + warp[1, 1] * source_y + warp[1, 2])
        * native.shape[0]
        / 256
    )
    left = int(round(center_x - PATCH / 2))
    top = int(round(center_y - PATCH / 2))
    left = int(np.clip(left, 0, native.shape[1] - PATCH))
    top = int(np.clip(top, 0, native.shape[0] - PATCH))

    # Same fixed post-crop warp and digital upsampling as a no-camera control.
    digital256 = cv2.warpAffine(
        source_bgr.astype(np.float32) / 255,
        warp,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    digital_native = cv2.resize(
        digital256, (native.shape[1], native.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )
    native_warp = warp.copy()
    native_warp[:2] *= native.shape[0] / 256
    controls = {"two_stage_lanczos": digital_native}
    controls["two_stage_lanczos_blurred"] = cv2.GaussianBlur(
        digital_native,
        (0, 0),
        sigmaX=0.7 * native.shape[0] / 256,
        borderType=cv2.BORDER_REFLECT,
    )
    for name, interpolation in (
        ("direct_linear", cv2.INTER_LINEAR),
        ("direct_cubic", cv2.INTER_CUBIC),
        ("direct_lanczos", cv2.INTER_LANCZOS4),
    ):
        controls[name] = cv2.warpAffine(
            source_bgr.astype(np.float32) / 255,
            native_warp,
            (native.shape[1], native.shape[0]),
            flags=interpolation,
            borderMode=cv2.BORDER_REFLECT,
        )
    crop = lambda image: image[top : top + PATCH, left : left + PATCH]

    def publish_rgb8(image: np.ndarray) -> np.ndarray:
        # True published PNGs are 8-bit. A floating-point control would have
        # an artificially low high-frequency floor after digital upsampling.
        bgr = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

    return crop(native), {
        name: publish_rgb8(crop(image)) for name, image in controls.items()
    }


def spectrum(
    patch: np.ndarray, ring_indices: np.ndarray, valid: np.ndarray, window: np.ndarray
) -> tuple[np.ndarray, float]:
    highpass = patch - cv2.GaussianBlur(
        patch, (0, 0), sigmaX=4, borderType=cv2.BORDER_REFLECT
    )
    rms = float(np.sqrt(np.mean(np.square(highpass))))
    power = np.square(np.abs(np.fft.fftshift(np.fft.fft2(highpass * window))))
    logpower = np.log10(power + 1e-10)
    whitened = np.zeros_like(logpower, dtype=np.float32)
    for ring in np.unique(ring_indices[valid]):
        mask = valid & (ring_indices == ring)
        whitened[mask] = logpower[mask] - np.median(logpower[mask])
    return whitened, rms


def main() -> None:
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    calibration = [row for row in rows if row["split"] == "calibration"]
    development = [row for row in rows if row["split"] == "development"]
    if len(calibration) != 240 or len(development) != 120:
        raise RuntimeError("expected frozen 240/120 nonreserved source groups")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    frequencies = np.fft.fftshift(np.fft.fftfreq(PATCH))
    fx, fy = np.meshgrid(frequencies, frequencies)
    radius = np.hypot(fx, fy)
    valid = (radius >= 0.08) & (radius <= 0.45) & ((fx > 0) | ((fx == 0) & (fy > 0)))
    ring_indices = np.floor(radius / 0.025).astype(np.int16)
    window = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)
    result = {
        "scope": "native published recapture resolution; frozen source-level calibration/development",
        "negative_control": "same original digitally warped/upscaled with five interpolation-and-blur variants, quantized to 8-bit, without display or camera",
        "warning": "calibration peak is a capture/publication periodicity candidate, not screen-subpixel ground truth",
        "conditions": {},
    }
    for condition in ("recap_mac", "recap_monitor"):
        warp = np.asarray(
            geometry["conditions"][condition]["median_calibration_affine"],
            dtype=np.float32,
        )
        calibration_map = np.zeros((PATCH, PATCH), dtype=np.float64)
        cal_native_rms = []
        for row in calibration:
            real_patch, _ = native_patches(row["src"], condition, warp)
            real_map, real_rms = spectrum(real_patch, ring_indices, valid, window)
            calibration_map += real_map
            cal_native_rms.append(real_rms)
        calibration_map /= len(calibration)
        candidate_map = np.where(valid, calibration_map, -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(candidate_map)), candidate_map.shape)
        candidate = (float(fx[iy, ix]), float(fy[iy, ix]))
        neighborhood = np.zeros_like(valid)
        neighborhood[max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = True
        neighborhood &= valid
        dev_native_peak = []
        dev_native_rms = []
        controls = {
            name: {"peak": [], "rms": []}
            for name in (
                "two_stage_lanczos",
                "two_stage_lanczos_blurred",
                "direct_linear",
                "direct_cubic",
                "direct_lanczos",
            )
        }
        for row in development:
            real_patch, control_patches = native_patches(row["src"], condition, warp)
            real_map, real_rms = spectrum(real_patch, ring_indices, valid, window)
            dev_native_peak.append(float(np.max(real_map[neighborhood])))
            dev_native_rms.append(real_rms)
            for name, patch in control_patches.items():
                control_map, control_rms = spectrum(patch, ring_indices, valid, window)
                controls[name]["peak"].append(float(np.max(control_map[neighborhood])))
                controls[name]["rms"].append(control_rms)
        result["conditions"][condition] = {
            "calibration_sources": len(calibration),
            "development_sources": len(development),
            "candidate_frequency_xy_cycles_per_native_pixel": candidate,
            "calibration_mean_whitened_log10_power_at_candidate": float(
                calibration_map[iy, ix]
            ),
            "calibration_native_highpass_rms": median_summary(cal_native_rms),
            "development_native_candidate_peak_log10_above_ring": median_summary(
                dev_native_peak
            ),
            "development_native_highpass_rms": median_summary(dev_native_rms),
            "digital_controls": {
                name: {
                    "candidate_peak_log10_above_ring": median_summary(measures["peak"]),
                    "highpass_rms": median_summary(measures["rms"]),
                    "native_peak_exceeds_control_count": int(
                        (
                            np.asarray(dev_native_peak) > np.asarray(measures["peak"])
                        ).sum()
                    ),
                    "native_rms_exceeds_control_count": int(
                        (np.asarray(dev_native_rms) > np.asarray(measures["rms"])).sum()
                    ),
                }
                for name, measures in controls.items()
            },
        }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
