"""Falsify an additive projector-channel/surface-mixing hypothesis.

Only 13 uniform charts are fitted: black and 4 nonzero levels for each of
red, green, blue. The remaining 112 chart colors and 200 textured test images
are predictions. Camera sRGB decoding is only a proxy for radiance; published
CompenNet PNGs do not establish the true camera response.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from work.probe_projector_chart_model import ARCHIVE, AUDIT, LEVELS, load_rgb, score


OUT = Path(__file__).with_name("projector_channel_basis_probe.json")
SIGMAS = (0.0, 0.45, 0.9, 1.35)


def to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def ref_index(channel: int, level: int) -> int:
    return 1 + level * (5 ** channel)


def fit(archive: zipfile.ZipFile, setup: str, linear_light: bool) -> tuple:
    prefix = f"{setup}/cam/warp/ref/"
    convert = to_linear if linear_light else lambda image: image
    black = convert(load_rgb(archive, prefix + "img_0001.png"))
    matrix = np.empty((256, 256, 3, 3), np.float32)
    tone = np.zeros((3, 5), np.float32)
    for channel in range(3):
        full = convert(load_rgb(archive, prefix + f"img_{ref_index(channel, 4):04d}.png"))
        vector = full - black
        matrix[..., channel] = vector
        denominator = float(np.sum(vector * vector))
        if denominator <= 1e-5:
            raise ValueError(f"projector channel {channel} has zero observed response")
        for level in range(1, 4):
            partial = convert(load_rgb(archive, prefix +
                                       f"img_{ref_index(channel, level):04d}.png"))
            tone[channel, level] = float(np.sum(vector * (partial - black)) / denominator)
        tone[channel, 4] = 1.0
    return black, matrix, tone, linear_light


def predict(source: np.ndarray, params: tuple, sigma: float) -> np.ndarray:
    black, matrix, tone, linear_light = params
    if sigma:
        source = gaussian_filter(source, (sigma, sigma, 0), mode="reflect")
    projected = np.stack([np.interp(source[..., c], LEVELS, tone[c])
                          for c in range(3)], axis=-1).astype(np.float32)
    mixed = black + np.einsum("hwoc,hwc->hwo", matrix, projected)
    return to_srgb(mixed) if linear_light else np.clip(mixed, 0, 1)


def run_setup(archive: zipfile.ZipFile, setup: str) -> dict:
    results = {}
    prefix = f"{setup}/cam/warp/"
    calibration_indices = {ref_index(channel, level)
                           for channel in range(3) for level in range(5)}
    assert len(calibration_indices) == 13
    for mode, linear in (("gamma_encoded", False), ("srgb_linearized", True)):
        params = fit(archive, setup, linear)
        mixed_chart_errors = []
        for index in range(1, 126):
            if index in calibration_indices:
                continue
            x = load_rgb(archive, f"ref/img_{index:04d}.png")
            y = load_rgb(archive, prefix + f"ref/img_{index:04d}.png")
            mixed_chart_errors.append(score(predict(x, params, 0), y)["mae"])
        train_mae = {str(sigma): [] for sigma in SIGMAS}
        for index in range(1, 17):
            x = load_rgb(archive, f"train/img_{index:04d}.png")
            y = load_rgb(archive, prefix + f"train/img_{index:04d}.png")
            for sigma in SIGMAS:
                train_mae[str(sigma)].append(score(predict(x, params, sigma), y)["mae"])
        chosen = min(SIGMAS, key=lambda sigma: np.mean(train_mae[str(sigma)]))
        test_mae = []
        for index in range(1, 201):
            x = load_rgb(archive, f"test/img_{index:04d}.png")
            y = load_rgb(archive, prefix + f"test/img_{index:04d}.png")
            test_mae.append(score(predict(x, params, chosen), y)["mae"])
        results[mode] = {"calibration_charts": 13, "unseen_mixed_charts": 112,
                         "mixed_chart_mae": float(np.mean(mixed_chart_errors)),
                         "train_sigma_images": 16, "chosen_sigma": chosen,
                         "train_mae_by_sigma": {s: float(np.mean(v)) for s, v in train_mae.items()},
                         "test_count": 200, "test_mae": float(np.mean(test_mae)),
                         "test_mae_by_image": test_mae,
                         "parameter_bytes_float32": int(params[0].nbytes + params[1].nbytes +
                                                        params[2].nbytes)}
    return {"setup": setup, "modes": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    setups = sorted(audit["setups"]) if args.all else ["light1/pos1/stripes",
                                                       "light2/pos1/curves",
                                                       "light3/pos3/squares",
                                                       "light4/pos2/curves"]
    if not audit["all_member_crc_passed"]:
        raise ValueError("full archive audit not passed")
    rows = []
    started = time.perf_counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for setup in setups:
            result = run_setup(archive, setup)
            rows.append(result)
            print(setup, {k: (round(v["mixed_chart_mae"], 4),
                              round(v["test_mae"], 4)) for k, v in result["modes"].items()}, flush=True)
            OUT.write_text(json.dumps({"archive_sha256": audit["archive_sha256"],
                                       "elapsed_sec": time.perf_counter() - started,
                                       "results": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
