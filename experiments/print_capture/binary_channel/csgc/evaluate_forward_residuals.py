"""Probe held-out CSGC residual structure of the fixed forward prototype."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import io
import json
import zipfile

import numpy as np
from PIL import Image

from palimpsest.simulation.print_scan import PrintScanParameters, simulate_print_scan


ARCHIVE = DATA_ROOT / "raw/csgc/CSGC_scan2400spi.zip"
OUT = WORK_DIR / "csgc_binary_forward_residuals.json"


def main() -> None:
    params_json = json.loads(
        (WORK_DIR / "csgc_binary_forward_holdout.json").read_text(encoding="utf-8")
    )
    params = PrintScanParameters(**params_json["params"])

    phase_sum = np.zeros((4, 4), dtype=np.float64)
    phase_sq = np.zeros((4, 4), dtype=np.float64)
    phase_n = np.zeros((4, 4), dtype=np.int64)
    bit_sum = np.zeros(2, dtype=np.float64)
    bit_sq = np.zeros(2, dtype=np.float64)
    bit_n = np.zeros(2, dtype=np.int64)
    image_mean_error = []

    with zipfile.ZipFile(ARCHIVE) as z:
        for item in range(101, 951):
            name = f"Scan{item:05d}.tif"
            with Image.open(
                io.BytesIO(z.read(f"2400dpi_NEW/bin_exact_crop_4/{name}"))
            ) as im:
                ref400 = np.asarray(im, dtype=np.uint8)
            with Image.open(
                io.BytesIO(z.read(f"2400dpi_NEW/resize_exact_crop/{name}"))
            ) as im:
                real = np.asarray(im, dtype=np.float32)[8:-8, 8:-8] / 255
            pred = simulate_print_scan(ref400[::4, ::4], params).scanner_output[
                8:-8, 8:-8
            ]
            err = pred - real
            image_mean_error.append(float(err.mean()))
            for py in range(4):
                for px in range(4):
                    values = err[py::4, px::4]
                    phase_sum[py, px] += values.sum(dtype=np.float64)
                    phase_sq[py, px] += np.sum(values * values, dtype=np.float64)
                    phase_n[py, px] += values.size
            binary = ref400[8:-8, 8:-8] > 127
            for bit in (0, 1):
                values = err[binary == bool(bit)]
                bit_sum[bit] += values.sum(dtype=np.float64)
                bit_sq[bit] += np.sum(values * values, dtype=np.float64)
                bit_n[bit] += values.size

    phase_mean = phase_sum / phase_n
    phase_rmse = np.sqrt(phase_sq / phase_n)
    report = {
        "n_holdout_codes": 850,
        "error": "simulated minus real, values normalized to [0,1]",
        "phase_mean_4x4": phase_mean.tolist(),
        "phase_rmse_4x4": phase_rmse.tolist(),
        "phase_mean_range": [float(phase_mean.min()), float(phase_mean.max())],
        "reference_bit_conditional": {
            "black_0": {
                "mean_error": float(bit_sum[0] / bit_n[0]),
                "rmse": float(np.sqrt(bit_sq[0] / bit_n[0])),
            },
            "white_1": {
                "mean_error": float(bit_sum[1] / bit_n[1]),
                "rmse": float(np.sqrt(bit_sq[1] / bit_n[1])),
            },
        },
        "per_code_mean_error_quantiles": {
            str(q): float(np.quantile(image_mean_error, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
