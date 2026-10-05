"""Run physical-coordinate binary print-scan prototype on 850 held-out codes.

The blur width and power exponent come from the prior 100-code *composite*
diagnostic. Assigning all blur to scanner_optical_sigma_um is a numerical
parameterization for this comparison, not an identification of that scanner.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import io
import json
import time
import zipfile

import numpy as np
from PIL import Image

from palimpsest.simulation.print_scan.monochrome import PrintScanParameters, simulate_print_scan


ARCHIVE = DATA_ROOT / "raw/csgc/CSGC_scan2400spi.zip"
OUT = WORK_DIR / "csgc_binary_forward_holdout.json"


def main() -> None:
    cal = json.loads(
        (WORK_DIR / "csgc_bounded_power_tone.json").read_text(encoding="utf-8")
    )
    exponent = cal["parameters"]["exponent"]
    effective_sigma_um = 2.5 * 25_400 / 2400
    aperture_sigma_um = 25_400 / (2400 * np.sqrt(12))
    optical_proxy_um = float(np.sqrt(effective_sigma_um**2 - aperture_sigma_um**2))
    params = PrintScanParameters(
        raster_mode="binary_direct",
        digital_ppi=600,
        render_ppi=4800,
        scan_ppi=2400,
        mechanical_dot_gain_um=0,
        deposition_blur_um=0,
        paper_scatter_sigma_um=0,
        scanner_optical_sigma_um=optical_proxy_um,
        paper_reflectance=1,
        ink_reflectance=0,
        scanner_gamma=1 / exponent,
    )

    with zipfile.ZipFile(ARCHIVE) as z:
        rows = []
        for item in range(101, 951):
            start = time.perf_counter()
            name = f"Scan{item:05d}.tif"
            with Image.open(
                io.BytesIO(z.read(f"2400dpi_NEW/bin_exact_crop_4/{name}"))
            ) as image:
                digital = np.asarray(image, dtype=np.uint8)[::4, ::4]
            with Image.open(
                io.BytesIO(z.read(f"2400dpi_NEW/resize_exact_crop/{name}"))
            ) as image:
                real = np.asarray(image, dtype=np.float32)[8:-8, 8:-8] / 255
            result = simulate_print_scan(digital, params)
            predicted = result.scanner_output[8:-8, 8:-8]
            if predicted.shape != real.shape:
                raise ValueError(f"shape mismatch at {name}")
            rows.append(
                {
                    "id": item,
                    "mae": float(np.mean(np.abs(predicted - real))),
                    "rmse": float(np.sqrt(np.mean((predicted - real) ** 2))),
                    "pearson": float(
                        np.corrcoef(predicted.ravel(), real.ravel())[0, 1]
                    ),
                    "end_to_end_decode_simulate_score_s": time.perf_counter() - start,
                }
            )

    def quantiles(key: str) -> dict:
        values = [row[key] for row in rows]
        return {
            str(q): float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        }

    report = {
        "calibration_ids": "00001-00100",
        "holdout_ids": "00101-00950",
        "n": len(rows),
        "parameter_interpretation": "composite fit mapped to one convenient stage; optical_proxy_um is not a measured scanner PSF",
        "effective_composite_sigma_um": effective_sigma_um,
        "scanner_aperture_approx_sigma_um": aperture_sigma_um,
        "optical_proxy_um": optical_proxy_um,
        "composite_power_exponent": exponent,
        "params": vars(params),
        "holdout_quantiles": {
            key: quantiles(key)
            for key in ("mae", "rmse", "pearson", "end_to_end_decode_simulate_score_s")
        },
        "per_template": rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "n",
                    "effective_composite_sigma_um",
                    "optical_proxy_um",
                    "composite_power_exponent",
                    "holdout_quantiles",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
