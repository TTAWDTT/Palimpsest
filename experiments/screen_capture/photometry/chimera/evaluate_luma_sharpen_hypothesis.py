"""Falsify simple post-tone luma sharpening against native Chimera recaptures.

Only one fixed output-space sigma and a tiny amount grid are tried. Amount is
selected on calibration luma band power; development tests chroma/coherence
and is never used to choose a parameter. This is not camera ISP calibration.
"""

from palimpsest.paths import WORK_DIR

import csv
import json

import cv2
import numpy as np

from experiments.screen_capture.spectral_structure.chimera.run_native_channel_coherence import (
    metrics,
)
from experiments.screen_capture.spectral_structure.chimera.protocol import (
    GEOMETRY,
    PATCH,
    ROOT,
    SPLIT,
    flattest_source_center,
)


OUTPUT = WORK_DIR / "chimera_luma_sharpen_hypothesis.json"
AMOUNTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
SIGMA = 1.0  # Native published-output pixels; preselected, not lens PSF.


def paired_patches(
    source_id: str, condition: str, warp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = cv2.imread(str(ROOT / "stylegan2_orig" / source_id), cv2.IMREAD_COLOR)
    real = cv2.imread(str(ROOT / condition / source_id), cv2.IMREAD_COLOR)
    if source is None or real is None or source.shape[:2] != (256, 256):
        raise RuntimeError(f"invalid pair: {source_id} {condition}")
    expected = 1026 if condition == "recap_mac" else 765
    if real.shape[:2] != (expected, expected):
        raise RuntimeError(f"native shape changed: {source_id} {condition}")
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    x, y = flattest_source_center(gray)
    scale = expected / 256
    cx = (warp[0, 0] * x + warp[0, 1] * y + warp[0, 2]) * scale
    cy = (warp[1, 0] * x + warp[1, 1] * y + warp[1, 2]) * scale
    left = int(np.clip(round(cx - PATCH / 2), 0, expected - PATCH))
    top = int(np.clip(round(cy - PATCH / 2), 0, expected - PATCH))
    crop = lambda image: image[top : top + PATCH, left : left + PATCH]
    digital256 = cv2.warpAffine(
        source.astype(np.float32) / 255,
        warp,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    digital_native = cv2.resize(
        digital256, (expected, expected), interpolation=cv2.INTER_LANCZOS4
    )
    return crop(real), crop(digital_native)


def sharpen_bgr(encoded_bgr: np.ndarray, amount: float) -> tuple[np.ndarray, float]:
    luma = (
        encoded_bgr[..., 0] * 0.0722
        + encoded_bgr[..., 1] * 0.7152
        + encoded_bgr[..., 2] * 0.2126
    )
    delta = amount * (
        luma
        - cv2.GaussianBlur(luma, (0, 0), sigmaX=SIGMA, borderType=cv2.BORDER_REFLECT)
    )
    adjusted = encoded_bgr + delta[..., None]
    clipped_fraction = float(((adjusted < 0) | (adjusted > 1)).mean())
    return np.rint(np.clip(adjusted, 0, 1) * 255).astype(np.uint8), clipped_fraction


def summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {
        "median": float(np.median(a)),
        "mean": float(a.mean()),
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
        raise RuntimeError("expected frozen 240/120 nonreserved groups")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    f = np.fft.fftshift(np.fft.fftfreq(PATCH))
    fx, fy = np.meshgrid(f, f)
    radius = np.hypot(fx, fy)
    mask = (radius >= 0.30) & (radius < 0.45)
    window = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)
    output = {
        "scope": "published-output negative control, not calibrated ISP",
        "candidate_amounts": AMOUNTS,
        "fixed_sigma_native_pixels": SIGMA,
        "selection": "calibration mean absolute luma highband dB error only",
        "conditions": {},
    }
    for condition in ("recap_mac", "recap_monitor"):
        warp = np.asarray(
            geometry["conditions"][condition]["median_calibration_affine"],
            dtype=np.float32,
        )
        records = []
        for row in rows:
            real, control_float = paired_patches(row["src"], condition, warp)
            actual = metrics(real, mask, window)
            candidates = {}
            for amount in AMOUNTS:
                generated, clip_fraction = sharpen_bgr(control_float, amount)
                candidates[amount] = {
                    **metrics(generated, mask, window),
                    "rgb_mae": float(
                        np.abs(
                            generated[16:-16, 16:-16].astype(np.float32)
                            - real[16:-16, 16:-16].astype(np.float32)
                        ).mean()
                        / 255
                    ),
                    "clip_fraction": clip_fraction,
                }
            records.append(
                {"split": row["split"], "actual": actual, "candidates": candidates}
            )
        calibration = [row for row in records if row["split"] == "calibration"]
        development = [row for row in records if row["split"] == "development"]
        calibration_luma_errors = {
            amount: [
                abs(
                    10
                    * np.log10(
                        (row["candidates"][amount]["luma_power"] + 1e-20)
                        / (row["actual"]["luma_power"] + 1e-20)
                    )
                )
                for row in calibration
            ]
            for amount in AMOUNTS
        }
        selected = min(
            AMOUNTS, key=lambda amount: np.mean(calibration_luma_errors[amount])
        )
        diagnostics = {}
        for amount in (0.0, selected):
            group = {}
            for metric in ("luma_power", "chroma_power", "mean_pair_coherence"):
                if metric.endswith("power"):
                    errors = [
                        abs(
                            10
                            * np.log10(
                                (row["candidates"][amount][metric] + 1e-20)
                                / (row["actual"][metric] + 1e-20)
                            )
                        )
                        for row in development
                    ]
                else:
                    errors = [
                        abs(row["candidates"][amount][metric] - row["actual"][metric])
                        for row in development
                    ]
                group[f"absolute_{metric}_error"] = summary(errors)
            group["rgb_mae"] = summary(
                [row["candidates"][amount]["rgb_mae"] for row in development]
            )
            group["clip_fraction"] = summary(
                [row["candidates"][amount]["clip_fraction"] for row in development]
            )
            diagnostics[str(amount)] = group
        output["conditions"][condition] = {
            "calibration_sources": len(calibration),
            "development_sources": len(development),
            "calibration_mean_abs_luma_db_by_amount": {
                str(amount): float(np.mean(values))
                for amount, values in calibration_luma_errors.items()
            },
            "selected_amount": selected,
            "development": diagnostics,
        }
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
