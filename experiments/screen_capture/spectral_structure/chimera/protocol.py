"""Search for repeatable native-pixel spectral peaks in true screen recaptures.

Calibration identifies a candidate frequency. Development tests that *fixed*
frequency against a geometry/resize-only digital control. A peak is not proof
of display pixels: CFA, ISP and publication sampling remain alternative causes.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

from pathlib import Path

import cv2
import numpy as np


ROOT = DATA_ROOT / "derived/chimera_paired"
SPLIT = DATA_ROOT / "manifests/chimera_simulation_source_split.csv"
GEOMETRY = WORK_DIR / "chimera_fixed_publication_geometry.json"
OUTPUT = WORK_DIR / "chimera_native_frequency_probe.json"
PATCH = 160


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
