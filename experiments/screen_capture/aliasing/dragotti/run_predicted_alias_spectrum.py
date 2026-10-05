"""Exploratory screen-grid alias-band probe on verified real recaptures.

This tests for narrow directional power near the *conditional* folded screen
frequency in selected smooth patches, with geometrically warped originals as
content controls. It cannot identify the actual camera sensor grid or prove
absence of moire; publication resizing and unknown display frame remain.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.screen_capture.geometry.dragotti.audit_pairs import align, read_rgb


AUDIT = Path("outputs/03_过程模拟/02_拍屏实测/Dragotti跨场景几何复核_2026-09-24.json")
OUT = WORK_DIR / "dragotti_predicted_alias_spectrum.json"
PATCH = 256


def alias_ratio(patch: np.ndarray, alias_frequency: float, axis: str) -> float:
    gray = np.asarray(patch, np.float32) / 255
    residual = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    window = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(residual * window))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(PATCH))
    fx, fy = np.meshgrid(freq, freq)
    along, across = (fx, fy) if axis == "horizontal" else (fy, fx)
    candidate = (np.abs(along - alias_frequency) <= 0.012) & (np.abs(across) <= 0.012)
    control = (
        (along >= 0.16)
        & (along <= 0.34)
        & (np.abs(along - alias_frequency) >= 0.03)
        & (np.abs(across) <= 0.012)
    )
    return float(
        np.quantile(power[candidate], 0.95) / max(np.median(power[control]), 1e-12)
    )


def sample_patches(
    warped_source: np.ndarray, target: np.ndarray
) -> list[tuple[int, int]]:
    source_gray = cv2.cvtColor(warped_source, cv2.COLOR_RGB2GRAY)
    smoothed = cv2.GaussianBlur(source_gray, (0, 0), sigmaX=2)
    rows = []
    for top in range(64, target.shape[0] - PATCH - 64, PATCH):
        for left in range(64, target.shape[1] - PATCH - 64, PATCH):
            tile = smoothed[top : top + PATCH, left : left + PATCH]
            score = float(cv2.Laplacian(tile, cv2.CV_32F).var())
            rows.append((score, top, left))
    return [(top, left) for _, top, left in sorted(rows)[:8]]


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = []
    for sample in audit["samples"]:
        source_path = Path(sample["original"]["file"])
        if (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            != sample["original"]["sha256"]
        ):
            raise RuntimeError("source SHA differs from audited archive member")
        source = read_rgb(source_path)
        for record in sample["recaptured"]:
            if record["camera"] not in ("EOS600D", "D3200"):
                continue
            target_path = Path(record["file"])
            if hashlib.sha256(target_path.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError("recapture SHA differs from audited archive member")
            target = read_rgb(target_path)
            transform, fit = align(source, target)
            warped = cv2.warpPerspective(
                source,
                transform,
                (target.shape[1], target.shape[0]),
                flags=cv2.INTER_CUBIC,
            )
            picks = sample_patches(warped, target)
            pitch = record["conditional_projected_screen_pitch_if_fit_1080_rows"]
            alias = abs(1 - 1 / pitch)
            comparisons = []
            for top, left in picks:
                src_patch = warped[top : top + PATCH, left : left + PATCH]
                dst_patch = target[top : top + PATCH, left : left + PATCH]
                for channel, color in enumerate("RGB"):
                    for axis in ("horizontal", "vertical"):
                        comparisons.append(
                            {
                                "top": top,
                                "left": left,
                                "color": color,
                                "axis": axis,
                                "recaptured_ratio": alias_ratio(
                                    dst_patch[..., channel], alias, axis
                                ),
                                "warped_original_ratio": alias_ratio(
                                    src_patch[..., channel], alias, axis
                                ),
                            }
                        )
            rows.append(
                {
                    "source_key": sample["source_key"],
                    "camera": record["camera"],
                    "conditional_pitch": pitch,
                    "predicted_alias_frequency": alias,
                    "fit_inliers": fit["ransac_inliers"],
                    "patch_count": len(picks),
                    "recaptured_ratio_median": float(
                        np.median([item["recaptured_ratio"] for item in comparisons])
                    ),
                    "warped_original_ratio_median": float(
                        np.median(
                            [item["warped_original_ratio"] for item in comparisons]
                        )
                    ),
                    "comparisons": comparisons,
                }
            )
            print(
                rows[-1]["source_key"],
                rows[-1]["camera"],
                rows[-1]["recaptured_ratio_median"],
                rows[-1]["warped_original_ratio_median"],
                flush=True,
            )
    output = {
        "method": "smoothest 8 256px tiles; highpass sigma3; 95th percentile alias-band power / median sideband power, per channel and axis",
        "conditional_alias_frequency_only": True,
        "rows": rows,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
