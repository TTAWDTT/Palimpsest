"""Fixed-parameter CSGC forward transfer on 12 cross-SPI digital templates."""

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from origin_simulation.print_scan import PrintScanParameters, simulate_print_scan


PROBE = Path(r"E:\ai_image_origin_research\data\derived\csgc_probe")
ARCHIVE = Path(r"E:\ai_image_origin_research\data\raw\csgc\CSGC_scan2400spi.zip")
OUT = Path("work/csgc_cross_spi_12_id_forward.json")
AUDIT = json.loads(
    Path("work/csgc_cross_spi_12_id_audit.json").read_text(encoding="utf-8")
)
IDS = AUDIT["ids_preselected"]
PARAMS = PrintScanParameters(
    **json.loads(
        Path("work/csgc_binary_forward_holdout.json").read_text(encoding="utf-8")
    )["params"]
)

rows = []
with zipfile.ZipFile(ARCHIVE) as z:
    for item in IDS:
        name = f"Scan{item:05d}.tif"
        with Image.open(
            io.BytesIO(z.read(f"2400dpi_NEW/bin_exact_crop_4/{name}"))
        ) as image:
            source = np.asarray(image, dtype=np.uint8)[::4, ::4]
        for spi, factor in ((2400, 4), (4800, 8), (9600, 16)):
            if spi == 2400:
                real_bytes = z.read(f"2400dpi_NEW/resize_exact_crop/{name}")
                with Image.open(io.BytesIO(real_bytes)) as image:
                    real = np.asarray(image, dtype=np.float32) / 255
            else:
                with Image.open(
                    PROBE / f"{spi}dpi_NEW__resize_exact_crop__{name}"
                ) as image:
                    real = np.asarray(image, dtype=np.float32) / 255
            pred = simulate_print_scan(
                source, replace(PARAMS, scan_ppi=spi, render_ppi=max(9600, spi))
            ).scanner_output
            margin = factor * 2
            p = pred[margin:-margin, margin:-margin]
            y = real[margin:-margin, margin:-margin]
            if p.shape != y.shape:
                raise RuntimeError(f"{spi}/{name}: mismatched shape")
            err = p - y
            rows.append(
                {
                    "id": item,
                    "spi": spi,
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "pearson": float(np.corrcoef(p.ravel(), y.ravel())[0, 1]),
                    "mean_error": float(np.mean(err)),
                }
            )


def metrics(spi: int) -> dict:
    subset = [r for r in rows if r["spi"] == spi]
    return {
        key: {
            str(q): float(np.quantile([r[key] for r in subset], q))
            for q in (0.05, 0.5, 0.95)
        }
        for key in ("mae", "rmse", "pearson", "mean_error")
    }


report = {
    "n_ids": len(IDS),
    "ids": IDS,
    "parameter_policy": "all parameters fixed from 2400-spi IDs 1-100; only scan PPI/render grid varied",
    "physical_paper_identity_verified": False,
    "rows": rows,
    "per_spi_quantiles": {str(spi): metrics(spi) for spi in (2400, 4800, 9600)},
}
OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report["per_spi_quantiles"], indent=2))
