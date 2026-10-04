"""Compare high-band RGB channel coherence in real recapture and digital controls.

Coherence is descriptive. Sensor, display, ISP and publication effects remain
entangled in the published RGB PNG. Reserved sources are not read.
"""

from palimpsest.paths import WORK_DIR

import csv
import json

import cv2
import numpy as np

from experiments.chimera.probe_chimera_native_frequency import (
    GEOMETRY,
    PATCH,
    ROOT,
    SPLIT,
    flattest_source_center,
)


OUTPUT = WORK_DIR / "chimera_native_channel_coherence.json"


def patches(source_id: str, condition: str, warp: np.ndarray) -> dict[str, np.ndarray]:
    source = cv2.imread(str(ROOT / "stylegan2_orig" / source_id), cv2.IMREAD_COLOR)
    native = cv2.imread(str(ROOT / condition / source_id), cv2.IMREAD_COLOR)
    if source is None or native is None or source.shape[:2] != (256, 256):
        raise RuntimeError(f"bad pair: {source_id} {condition}")
    expected = 1026 if condition == "recap_mac" else 765
    if native.shape[:2] != (expected, expected):
        raise RuntimeError(f"native shape changed: {source_id} {condition}")
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    x, y = flattest_source_center(gray)
    scale = expected / 256
    cx = (warp[0, 0] * x + warp[0, 1] * y + warp[0, 2]) * scale
    cy = (warp[1, 0] * x + warp[1, 1] * y + warp[1, 2]) * scale
    left = int(np.clip(round(cx - PATCH / 2), 0, expected - PATCH))
    top = int(np.clip(round(cy - PATCH / 2), 0, expected - PATCH))
    crop = lambda image: image[top : top + PATCH, left : left + PATCH]
    source_float = source.astype(np.float32) / 255
    digital256 = cv2.warpAffine(
        source_float,
        warp,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    two = cv2.resize(digital256, (expected, expected), interpolation=cv2.INTER_LANCZOS4)
    direct_warp = warp.copy()
    direct_warp[:2] *= scale
    direct = cv2.warpAffine(
        source_float,
        direct_warp,
        (expected, expected),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    quantize = lambda image: np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return {
        "real": crop(native),
        "two_stage_lanczos": quantize(crop(two)),
        "direct_linear": quantize(crop(direct)),
    }


def metrics(
    patch: np.ndarray, mask: np.ndarray, window: np.ndarray
) -> dict[str, float]:
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    fields = (rgb - rgb.mean(axis=(0, 1))) * window[..., None]
    spectrum = np.fft.fftshift(np.fft.fft2(fields, axes=(0, 1)), axes=(0, 1))[mask]
    auto = np.sum(np.square(np.abs(spectrum)), axis=0)
    coherence = []
    for a, b in ((0, 1), (1, 2), (0, 2)):
        cross = np.sum(spectrum[:, a] * np.conjugate(spectrum[:, b]))
        coherence.append(float(np.abs(cross) / np.sqrt(auto[a] * auto[b] + 1e-20)))
    luma = spectrum @ np.array([0.2126, 0.7152, 0.0722])
    chroma_rg = spectrum[:, 0] - spectrum[:, 1]
    chroma_bg = spectrum[:, 2] - spectrum[:, 1]
    luma_power = float(np.mean(np.abs(luma) ** 2))
    chroma_power = float(
        (np.mean(np.abs(chroma_rg) ** 2) + np.mean(np.abs(chroma_bg) ** 2)) / 2
    )
    return {
        "mean_pair_coherence": float(np.mean(coherence)),
        "rg_coherence": coherence[0],
        "gb_coherence": coherence[1],
        "rb_coherence": coherence[2],
        "luma_power": luma_power,
        "chroma_power": chroma_power,
        "chroma_to_luma_power": chroma_power / max(luma_power, 1e-20),
    }


def summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.1)),
        "p90": float(np.quantile(a, 0.9)),
    }


def main() -> None:
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["split"] in ("calibration", "development")
        ]
    if len(rows) != 360:
        raise RuntimeError("expected 360 nonreserved sources")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    f = np.fft.fftshift(np.fft.fftfreq(PATCH))
    fx, fy = np.meshgrid(f, f)
    radius = np.hypot(fx, fy)
    mask = (radius >= 0.30) & (radius < 0.45)
    window = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)
    result = {
        "scope": "native 8-bit published RGB, nonreserved source-level split",
        "warning": "coherence cannot uniquely identify sensor noise or display subpixels",
        "conditions": {},
    }
    for condition in ("recap_mac", "recap_monitor"):
        warp = np.asarray(
            geometry["conditions"][condition]["median_calibration_affine"],
            dtype=np.float32,
        )
        entries = []
        for row in rows:
            compared = {
                name: metrics(image, mask, window)
                for name, image in patches(row["src"], condition, warp).items()
            }
            entries.append(
                {
                    "split": row["split"],
                    "scene": row["scene"],
                    "label": row["label"],
                    "metrics": compared,
                }
            )
        result["conditions"][condition] = {}
        for split in ("calibration", "development"):
            group = [entry for entry in entries if entry["split"] == split]
            result["conditions"][condition][split] = {
                "count": len(group),
                "controls": {},
            }
            for name in ("two_stage_lanczos", "direct_linear"):
                control = {}
                for metric in (
                    "mean_pair_coherence",
                    "rg_coherence",
                    "gb_coherence",
                    "rb_coherence",
                    "chroma_to_luma_power",
                    "luma_power",
                    "chroma_power",
                ):
                    true_values = [entry["metrics"]["real"][metric] for entry in group]
                    digital_values = [entry["metrics"][name][metric] for entry in group]
                    diff = np.asarray(true_values) - np.asarray(digital_values)
                    control[metric] = {
                        "real": summary(true_values),
                        "digital": summary(digital_values),
                        "real_minus_digital": summary(diff.tolist()),
                        "real_exceeds_digital_count": int((diff > 0).sum()),
                    }
                    if metric in ("luma_power", "chroma_power"):
                        ratios = 10 * np.log10(
                            (np.asarray(true_values) + 1e-20)
                            / (np.asarray(digital_values) + 1e-20)
                        )
                        control[metric]["real_over_digital_db"] = summary(
                            ratios.tolist()
                        )
                result["conditions"][condition][split]["controls"][name] = control
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                condition: {
                    split: result["conditions"][condition][split]["controls"][
                        "two_stage_lanczos"
                    ]["mean_pair_coherence"]
                    for split in ("calibration", "development")
                }
                for condition in ("recap_mac", "recap_monitor")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
