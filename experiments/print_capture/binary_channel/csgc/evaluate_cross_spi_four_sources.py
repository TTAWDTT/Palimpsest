"""Small falsification probe: transfer 2400-spi composite parameters to other SPI.

Only four IDs were CRC-checked by HTTP Range for 4800/9600. The same digital
template is verified across SPIs; the same physical paper is NOT established.
This script does not tune parameters on these samples.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import json
from dataclasses import replace

import numpy as np
from PIL import Image

from palimpsest.simulation.print_scan.monochrome import PrintScanParameters, simulate_print_scan


PROBE = DATA_ROOT / "derived/csgc_probe"


def main() -> None:
    PARAMS = PrintScanParameters(
        **json.loads(
            (WORK_DIR / "csgc_binary_forward_holdout.json").read_text(encoding="utf-8")
        )["params"]
    )
    OUT = WORK_DIR / "csgc_cross_spi_forward_probe.json"
    IDS = (1, 101, 501, 901)
    SCALES = ((2400, 4), (4800, 8), (9600, 16))

    rows = []
    for spi, scale in SCALES:
        for item in IDS:
            prefix = f"{spi}dpi_NEW__"
            name = f"Scan{item:05d}.tif"
            with Image.open(PROBE / f"{prefix}bin_exact_crop_{scale}__{name}") as image:
                source = np.asarray(image, dtype=np.uint8)[::scale, ::scale]
            with Image.open(PROBE / f"{prefix}resize_exact_crop__{name}") as image:
                real = np.asarray(image, dtype=np.float32) / 255.0
            sim = simulate_print_scan(
                source, replace(PARAMS, scan_ppi=spi, render_ppi=max(9600, spi))
            ).scanner_output
            margin = 2 * scale
            pred = sim[margin:-margin, margin:-margin]
            target = real[margin:-margin, margin:-margin]
            if pred.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {spi} {name}: {pred.shape}, {target.shape}"
                )
            err = pred - target
            rows.append(
                {
                    "spi": spi,
                    "id": item,
                    "n_pixels": int(err.size),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "pearson": float(np.corrcoef(pred.ravel(), target.ravel())[0, 1]),
                    "mean_error": float(np.mean(err)),
                    "mean_real": float(np.mean(target)),
                    "mean_sim": float(np.mean(pred)),
                }
            )

    report = {
        "purpose": "transfer fixed 2400-spi composite parameters without retuning",
        "status": "4 digital IDs per SPI only; identical physical sheets across SPI unverified",
        "crop": "2 binary cells on each edge (8/16/32 output pixels)",
        "rows": rows,
        "per_spi_median": {
            str(spi): {
                k: float(np.median([r[k] for r in rows if r["spi"] == spi]))
                for k in ("mae", "rmse", "pearson", "mean_error")
            }
            for spi, _ in SCALES
        },
        "per_spi_holdout_median": {
            str(spi): {
                k: float(
                    np.median([r[k] for r in rows if r["spi"] == spi and r["id"] > 100])
                )
                for k in ("mae", "rmse", "pearson", "mean_error")
            }
            for spi, _ in SCALES
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["per_spi_holdout_median"], indent=2))


if __name__ == "__main__":
    main()
