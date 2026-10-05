"""Diagnostic of which fixed-input structures explain CSGC forward residuals.

Residual correction tables are fitted on IDs 1-100 only and evaluated on
101-950. They are descriptive, not a physically identified print simulator.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import io
import json
import zipfile

import numpy as np
from PIL import Image

from palimpsest.simulation.print_scan.monochrome import PrintScanParameters, simulate_print_scan


ARCHIVE = DATA_ROOT / "raw/csgc/CSGC_scan2400spi.zip"
OUT = WORK_DIR / "csgc_residual_conditional_diagnostic.json"


def main() -> None:
    PARAMS = PrintScanParameters(
        **json.loads(
            (WORK_DIR / "csgc_binary_forward_holdout.json").read_text(encoding="utf-8")
        )["params"]
    )
    INNER = np.s_[8:-8, 8:-8]
    GRID = np.indices((400, 400))
    PHASE = ((GRID[0] % 4) * 4 + GRID[1] % 4)[INNER]

    def codes(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bits = source > 127
        padded = np.pad(bits.astype(np.int16), 1, mode="edge")
        pattern = np.zeros((100, 100), dtype=np.int16)
        for dy in range(3):
            for dx in range(3):
                pattern += padded[dy : dy + 100, dx : dx + 100] * (1 << (3 * dy + dx))
        bit_group = (
            np.repeat(np.repeat(bits.astype(np.int16), 4, axis=0), 4, axis=1)[INNER]
            * 16
            + PHASE
        )
        neighborhood_group = (
            np.repeat(np.repeat(pattern, 4, axis=0), 4, axis=1)[INNER] * 16 + PHASE
        )
        return bit_group.ravel(), neighborhood_group.ravel()

    def pair(
        z: zipfile.ZipFile, item: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        name = f"Scan{item:05d}.tif"
        with Image.open(
            io.BytesIO(z.read(f"2400dpi_NEW/bin_exact_crop_4/{name}"))
        ) as im:
            source = np.asarray(im, dtype=np.uint8)[::4, ::4]
        with Image.open(
            io.BytesIO(z.read(f"2400dpi_NEW/resize_exact_crop/{name}"))
        ) as im:
            real = np.asarray(im, dtype=np.float32)[INNER] / 255
        predicted = simulate_print_scan(source, PARAMS).scanner_output[INNER]
        return source, predicted, real

    global_sum = 0.0
    global_n = 0
    per_code = []
    bit_sum = np.zeros(32, dtype=np.float64)
    bit_n = np.zeros(32, dtype=np.int64)
    local_sum = np.zeros(8192, dtype=np.float64)
    local_n = np.zeros(8192, dtype=np.int64)

    with zipfile.ZipFile(ARCHIVE) as z:
        for item in range(1, 101):
            source, pred, real = pair(z, item)
            err = (pred - real).ravel()
            per_code.append(
                {
                    "id": item,
                    "mean_error": float(np.mean(err)),
                    "source_white_fraction": float(np.mean(source > 127)),
                }
            )
            bit_group, local_group = codes(source)
            global_sum += err.sum(dtype=np.float64)
            global_n += err.size
            bit_sum += np.bincount(bit_group, weights=err, minlength=32)
            bit_n += np.bincount(bit_group, minlength=32)
            local_sum += np.bincount(local_group, weights=err, minlength=8192)
            local_n += np.bincount(local_group, minlength=8192)

    global_bias = global_sum / global_n
    bit_bias = np.divide(bit_sum, bit_n, out=np.full(32, global_bias), where=bit_n > 0)
    local_bias = np.divide(
        local_sum, local_n, out=np.full(8192, global_bias), where=local_n > 0
    )

    # Additive neighborhood model: a compact 3x3 linear response at each output
    # phase. A general 512-pattern table also captures interactions; the gap
    # between the two is a diagnostic, not a physical attribution.
    patterns = np.arange(512, dtype=np.int32)
    design = np.column_stack(
        [np.ones(512), np.stack([((patterns >> bit) & 1) for bit in range(9)], axis=1)]
    )
    linear_bias = np.empty(8192, dtype=np.float64)
    linear_coefficients = []
    for phase in range(16):
        index = patterns * 16 + phase
        weight = np.sqrt(local_n[index].astype(np.float64))
        beta = np.linalg.lstsq(
            design * weight[:, None], local_bias[index] * weight, rcond=None
        )[0]
        linear_bias[index] = design @ beta
        linear_coefficients.append(beta.tolist())

    scores = {
        name: []
        for name in (
            "uncorrected",
            "global_bias",
            "bit_phase",
            "linear3x3_phase",
            "local3x3_phase",
        )
    }
    with zipfile.ZipFile(ARCHIVE) as z:
        for item in range(101, 951):
            source, pred, real = pair(z, item)
            per_code.append(
                {
                    "id": item,
                    "mean_error": float(np.mean(pred - real)),
                    "source_white_fraction": float(np.mean(source > 127)),
                }
            )
            bit_group, local_group = codes(source)
            proposals = {
                "uncorrected": pred,
                "global_bias": np.clip(pred - global_bias, 0, 1),
                "bit_phase": np.clip(
                    pred - bit_bias[bit_group].reshape(pred.shape), 0, 1
                ),
                "linear3x3_phase": np.clip(
                    pred - linear_bias[local_group].reshape(pred.shape), 0, 1
                ),
                "local3x3_phase": np.clip(
                    pred - local_bias[local_group].reshape(pred.shape), 0, 1
                ),
            }
            for name, trial in proposals.items():
                err = trial - real
                scores[name].append(
                    {
                        "mae": float(np.mean(np.abs(err))),
                        "rmse": float(np.sqrt(np.mean(err**2))),
                        "pearson": float(
                            np.corrcoef(trial.ravel(), real.ravel())[0, 1]
                        ),
                    }
                )

    report = {
        "calibration_ids": "00001-00100",
        "holdout_ids": "00101-00950",
        "n_calibration_pixels": global_n,
        "n_holdout_images": 850,
        "global_calibration_bias": global_bias,
        "per_100_id_block": {
            f"{start:05d}-{min(start + 99, 950):05d}": {
                "n": len(
                    [r for r in per_code if start <= r["id"] <= min(start + 99, 950)]
                ),
                "median_mean_error": float(
                    np.median(
                        [
                            r["mean_error"]
                            for r in per_code
                            if start <= r["id"] <= min(start + 99, 950)
                        ]
                    )
                ),
                "median_white_fraction": float(
                    np.median(
                        [
                            r["source_white_fraction"]
                            for r in per_code
                            if start <= r["id"] <= min(start + 99, 950)
                        ]
                    )
                ),
            }
            for start in range(1, 951, 100)
        },
        "correlation_mean_error_vs_white_fraction": float(
            np.corrcoef(
                [r["mean_error"] for r in per_code],
                [r["source_white_fraction"] for r in per_code],
            )[0, 1]
        ),
        "per_code": per_code,
        "local_group_count_min": int(local_n.min()),
        "local_group_count_median": float(np.median(local_n)),
        "linear_model_coefficients_by_phase": linear_coefficients,
        "methods": {
            name: {
                metric: {
                    str(q): float(np.quantile([r[metric] for r in values], q))
                    for q in (0.05, 0.5, 0.95)
                }
                for metric in ("mae", "rmse", "pearson")
            }
            for name, values in scores.items()
        },
        "interpretation": "empirical correction tables are diagnostics, not identified physical stages",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("global_calibration_bias", "local_group_count_min", "methods")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
