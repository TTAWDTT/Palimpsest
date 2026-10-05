"""Check the prototype's internal fine-grid convergence on a virtual patch."""

from dataclasses import replace
import json

import numpy as np

from palimpsest.simulation.print_scan.monochrome import PrintScanParameters, simulate_print_scan


def main() -> None:
    digital = np.full((128, 128), 0.72, dtype=np.float32)
    base = PrintScanParameters(
        digital_ppi=800,
        render_ppi=1600,
        scan_ppi=800,
        screen_lpi=100,
        screen_angle_degrees=37,
        paper_scatter_sigma_um=15,
        scanner_optical_sigma_um=12,
    )
    outputs = {}
    for render_ppi in (1600, 2400, 3200, 4800):
        result = simulate_print_scan(digital, replace(base, render_ppi=render_ppi))
        outputs[render_ppi] = result.scanner_output
    high = outputs[4800]
    report = {
        str(ppi): {
            "mean": float(image.mean()),
            "std": float(image.std()),
            "mae_vs_4800": float(np.mean(np.abs(image - high))),
            "max_abs_vs_4800": float(np.max(np.abs(image - high))),
        }
        for ppi, image in outputs.items()
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
