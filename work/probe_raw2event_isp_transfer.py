"""Fit a minimal RAW-to-RGB color chain on one Raw2Event prefix; test on another.

This is a falsification probe, not an ISP reconstruction: it lacks black-level,
white-balance, lens shading, denoise, tone, and device metadata supervision.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from work.audit_raw2event_probe import ROOT, extract_frame, HEIGHT, WIDTH


AUTOMOBILE = "10000_automobile_5_1087_20251224_105416"
AIRPLANE = "1000_airplane_1_9934_20251222_161953"
AUDITS = {
    AUTOMOBILE: Path("work/raw2event_probe_pixel_audit.json"),
    AIRPLANE: Path("work/raw2event_probe_airplane_pixel_audit.json"),
}
OUT = Path("work/raw2event_isp_transfer.json")
VIEW = Path("work/raw2event_probe_airplane/isp_transfer")
PHASES = {
    "RG": cv2.COLOR_BayerRG2RGB,
    "BG": cv2.COLOR_BayerBG2RGB,
    "GR": cv2.COLOR_BayerGR2RGB,
    "GB": cv2.COLOR_BayerGB2RGB,
}


def srgb_decode(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def srgb_encode(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)


def load_pair(prefix: str, phase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = extract_frame(ROOT / "frames_raw" / f"{prefix}.mkv", 0, "gray16le", 1, "<u2")
    rgb = extract_frame(ROOT / "frames_rgb" / f"{prefix}.mkv", 0, "rgb24", 3, "u1").astype(np.float32) / 255
    audit = json.loads(AUDITS[prefix].read_text(encoding="utf-8"))
    h = np.asarray(audit["samples"]["0"]["tag"]["raw_to_rgb_tag_homography"], dtype=np.float64)
    demosaiced = cv2.cvtColor(raw, PHASES[phase]).astype(np.float32) / 1023
    demosaiced = cv2.GaussianBlur(demosaiced, (3, 3), 0.6)
    x = cv2.warpPerspective(demosaiced, h, (WIDTH, HEIGHT))
    valid = cv2.warpPerspective(np.ones_like(raw, dtype=np.uint8), h, (WIDTH, HEIGHT)) == 1
    valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return x, rgb, valid


def fit(x: np.ndarray, y: np.ndarray, mask: np.ndarray, target_space: str) -> np.ndarray:
    xx = np.column_stack((x[mask], np.ones(mask.sum(), dtype=np.float32)))
    yy = srgb_decode(y[mask]) if target_space == "linear_light" else y[mask]
    # Unconstrained 3x3 plus bias; regularized for invertibility, not positivity.
    gram = xx.T @ xx + np.diag([1e-4, 1e-4, 1e-4, 0])
    return np.linalg.solve(gram, xx.T @ yy)


def predict(x: np.ndarray, weights: np.ndarray, target_space: str) -> np.ndarray:
    mapped = x @ weights[:3] + weights[3]
    return np.clip(srgb_encode(mapped) if target_space == "linear_light" else mapped, 0, 1)


def measure(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    # The crop is a frozen approximate stimulus location in both first RGB frames.
    content = np.zeros(valid.shape, bool)
    content[185:365, 225:400] = True
    masks = {"all_valid": valid, "stimulus_crop": valid & content}
    return {name: {"n": int(mask.sum()),
                   "mae_255": float(np.abs(pred[mask] - truth[mask]).mean() * 255),
                   "channel_mae_255": [float(v) for v in np.abs(pred[mask] - truth[mask]).mean(axis=0) * 255]}
            for name, mask in masks.items()}


def main() -> None:
    all_rows = []
    best = None
    for phase in PHASES:
        x_cal, y_cal, m_cal = load_pair(AUTOMOBILE, phase)
        x_test, y_test, m_test = load_pair(AIRPLANE, phase)
        for space in ("encoded_rgb", "linear_light"):
            weights = fit(x_cal, y_cal, m_cal, space)
            cal_metrics = measure(predict(x_cal, weights, space), y_cal, m_cal)
            test_pred = predict(x_test, weights, space)
            test_metrics = measure(test_pred, y_test, m_test)
            row = {"phase": phase, "target_space": space,
                   "fit_prefix": AUTOMOBILE, "test_prefix": AIRPLANE,
                   "weights_4x3": weights.tolist(),
                   "calibration": cal_metrics, "heldout": test_metrics}
            all_rows.append(row)
            if best is None or cal_metrics["stimulus_crop"]["mae_255"] < best[0]:
                best = (cal_metrics["stimulus_crop"]["mae_255"], row, test_pred)
    assert best is not None
    VIEW.mkdir(parents=True, exist_ok=True)
    Image.fromarray((best[2] * 255).round().astype(np.uint8), "RGB").save(VIEW / "predicted_airplane_rgb.png")
    _, truth, _ = load_pair(AIRPLANE, best[1]["phase"])
    Image.fromarray((truth * 255).round().astype(np.uint8), "RGB").save(VIEW / "actual_airplane_rgb.png")
    by_key = {(row["phase"], row["target_space"]): row for row in all_rows}
    # With unconstrained 3x3 color mixing, swapping the demosaic R/B channels
    # can be absorbed by the fitted matrix. This cannot identify CFA phase.
    phase_tie_errors = {}
    for left, right in (("RG", "BG"), ("GR", "GB")):
        phase_tie_errors[f"{left}_vs_{right}"] = max(
            abs(by_key[(left, space)]["heldout"]["stimulus_crop"]["mae_255"]
                - by_key[(right, space)]["heldout"]["stimulus_crop"]["mae_255"])
            for space in ("encoded_rgb", "linear_light"))
    report = {"scope": "first frame of two content-distinct prefixes, geometry from each Tag frame0",
              "selection": "phase and target-space selected on automobile stimulus crop only",
              "phase_identifiability": "RG/BG and GR/GB are equivalent under unconstrained 3x3 color mapping",
              "phase_tie_heldout_stimulus_mae_255": phase_tie_errors,
              "best_on_calibration": {k: v for k, v in best[1].items() if k != "weights_4x3"},
              "all_candidates": all_rows,
              "limit": "one calibration recording; no independent device parameter labels, motion or exposure controls"}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["best_on_calibration"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
