"""Frozen M: two dimensionless increment-moment features, no learned encoder.

This extracts the marginal branch of the 2026-10-09 experiment without its
relation/null branches. Formula, preprocessing and calibration are unchanged.
Scores are evidence, not probabilities. Calibration used labeled originals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from palimpsest.contracts import Prediction, RGBImage, validate_rgb

SIDE = 256
BLOCK = 32
GRID = SIDE // BLOCK
SIGMAS = (1.0, 2.0, 4.0, 8.0)
DISTANCES = (1, 2, 4, 8)
AXES = (
    "intermittency_ess_deviation",
    "intermittency_logmoment_gaussian_offset",
)
GAUSSIAN_LOG_RATIO = float(np.log(3.0 * np.pi / 8.0))
LOG_DENOM_REL_TOL = 1e-4
# Prediction uses strict >. Its closest representable negative threshold
# preserves the published M>=0 rule, including +0.0 and -0.0.
LEGACY_THRESHOLD = float(np.nextafter(0.0, -np.inf))


def structure_functions(luminance: np.ndarray) -> np.ndarray:
    """S2/S3/S4 [64 blocks, 4 sigmas, 4 strides, 4 directions, 3 orders]."""
    y = np.asarray(luminance, dtype=np.float64)
    if y.shape != (SIDE, SIDE) or not np.isfinite(y).all():
        raise ValueError("Expected finite 256x256 luminance")
    curves = np.zeros((GRID * GRID, 4, 4, 4, 3), dtype=np.float64)
    for si, sigma in enumerate(SIGMAS):
        smoothed = gaussian_filter(y, sigma, mode="reflect", truncate=4.0)
        blocks = (
            smoothed.reshape(GRID, BLOCK, GRID, BLOCK)
            .transpose(0, 2, 1, 3)
            .reshape(GRID * GRID, BLOCK, BLOCK)
        )
        for ri, r in enumerate(DISTANCES):
            # Diagonal Euclidean distance is sqrt(2)*r. No wrap/cross-block pairs.
            for ti, (dy, dx) in enumerate(((0, r), (r, r), (r, 0), (r, -r))):
                y0, y1 = max(0, -dy), BLOCK - max(0, dy)
                x0, x1 = max(0, -dx), BLOCK - max(0, dx)
                inc = np.abs(
                    blocks[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                    - blocks[:, y0:y1, x0:x1]
                )
                for pi, order in enumerate((2, 3, 4)):
                    curves[:, si, ri, ti, pi] = np.mean(inc**order, axis=(1, 2))
    return curves


def extract_luminance(luminance: np.ndarray) -> dict:
    """Float path for the exact affine theorem (no quantization or clipping)."""
    y = np.asarray(luminance, dtype=np.float64)
    curves = structure_functions(y)
    positive = np.isfinite(curves) & (curves > 0.0)
    logs = np.full_like(curves, np.nan)
    logs[positive] = np.log(curves[positive])
    dr = np.diff(logs, axis=2)
    ends = positive[:, :, 1:, :, :] & positive[:, :, :-1, :, :]
    numerator, denominator = dr[..., 2], dr[..., 1]
    scale = np.maximum(1.0, np.maximum(np.abs(numerator), np.abs(denominator)))
    valid = (
        ends[..., 2] & ends[..., 1] & (np.abs(denominator) > LOG_DENOM_REL_TOL * scale)
    )
    ratios = numerator[valid] / denominator[valid]
    shape_valid = np.all(positive, axis=-1)
    shape = (
        logs[..., 2][shape_valid]
        + logs[..., 0][shape_valid]
        - 2.0 * logs[..., 1][shape_valid]
    )
    features = {
        AXES[0]: float(np.median(np.abs(ratios - 4.0 / 3.0))) if ratios.size else 0.0,
        AXES[1]: float(np.median(shape) - GAUSSIAN_LOG_RATIO) if shape.size else 0.0,
    }
    qc = {
        "ess_r_valid_fraction": float(np.mean(valid)),
        "shape_valid_fraction": float(np.mean(shape_valid)),
        "minimum_finite_logmoment_shape": float(np.min(shape)) if shape.size else None,
        "logmoment_cauchy_violations_below_minus_1e_10": int(np.sum(shape < -1e-10)),
        "input_variance": float(np.var(y)),
        "degenerate": bool(not ratios.size or not shape.size or np.var(y) == 0.0),
    }
    return {"features": features, "qc": qc}


def extract_features(image: RGBImage) -> dict:
    """Stored RGB orientation; direct Lanczos resize, no ICC/EXIF/gamma step."""
    validate_rgb(image)
    resized = Image.fromarray(image).resize((SIDE, SIDE), Image.Resampling.LANCZOS)
    rgb = np.asarray(resized, dtype=np.float64) / 255.0
    return extract_luminance(
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    )


@dataclass(frozen=True)
class AxisCalibration:
    center: float
    scale: float
    direction: float

    def __post_init__(self) -> None:
        if (
            not all(np.isfinite(v) for v in (self.center, self.scale, self.direction))
            or self.scale <= 0.0
            or self.direction not in (-1.0, 1.0)
        ):
            raise ValueError(
                "Invalid M calibration: finite center, positive scale, +/-1 direction required"
            )


class MIntermittencyDetector:
    name = "M marginal intermittency (frozen 2026-10-09)"

    def __init__(self, calibration: Path | None = None):
        payload = (
            Path(calibration).read_bytes()
            if calibration is not None
            else files("palimpsest.detection.algorithms")
            .joinpath("m_calibration.json")
            .read_bytes()
        )
        declared = json.loads(payload)
        if declared.get("schema") != 1 or set(declared.get("features", {})) != set(
            AXES
        ):
            raise ValueError(
                "Expected M calibration schema 1 with exactly two declared axes"
            )
        self.parameters = tuple(
            AxisCalibration(**declared["features"][axis]) for axis in AXES
        )
        self.calibration_sha256 = hashlib.sha256(payload).hexdigest()

    def score_features(
        self, features: dict[str, float]
    ) -> tuple[float, dict[str, float]]:
        if set(features) != set(AXES) or not all(
            np.isfinite(v) for v in features.values()
        ):
            raise ValueError("Expected two finite M features")
        standardized = {
            axis: float(p.direction * (features[axis] - p.center) / p.scale)
            for axis, p in zip(AXES, self.parameters)
        }
        score = float(np.median(list(standardized.values())))
        if not np.isfinite(score):
            raise ValueError("Nonfinite M score")
        return score, standardized

    def analyze(self, image: RGBImage) -> dict:
        start = perf_counter()
        result = extract_features(image)
        score, standardized = self.score_features(result["features"])
        return {
            **result,
            "standardized_features": standardized,
            "score": score,
            "prediction": int(score >= 0.0),
            "calibration_sha256": self.calibration_sha256,
            "timing_ms": {"preprocess_and_features": (perf_counter() - start) * 1000},
        }

    def predict(self, image: RGBImage) -> Prediction:
        result = self.analyze(image)
        return Prediction(
            method=self.name,
            score=result["score"],
            threshold=LEGACY_THRESHOLD,
            score_kind="standardized_statistical_evidence",
            timing_ms=result["timing_ms"],
            metadata={
                "features": json.dumps(result["features"], allow_nan=False),
                "standardized_features": json.dumps(
                    result["standardized_features"], allow_nan=False
                ),
                "qc": json.dumps(result["qc"], allow_nan=False),
                "calibration_sha256": self.calibration_sha256,
                "decision_rule": "M>=0 AI; M<0 natural; QC-degenerate forced output is not reliable evidence",
                "calibration_scope": "738 Chimera source originals; labeled robust calibration; no learned encoder",
            },
        )
