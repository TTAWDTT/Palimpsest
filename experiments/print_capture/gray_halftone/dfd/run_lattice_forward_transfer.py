"""Calibrate one effective square halftone lattice on D5; predict D6 peaks.

The existing RGB->CMYK->paper->scanner forward chain supplies spatial order.
This is an *effective frequency hypothesis*: DFD has no digital target/RIP
record to prove that the observed peaks are the true printer screen basis.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import json

import numpy as np

from palimpsest.simulation.color_print_scan import (
    ColorPrintScanParameters,
    simulate_color_print_scan,
)
from experiments.print_capture.color_halftone.dfd.run_color_matched_printers import (
    top_spectral_peaks,
)


MATCHED = WORK_DIR / "dfd_color_matched_printers.json"
OUT = WORK_DIR / "dfd_lattice_forward_transfer.json"


def xy(row: dict) -> np.ndarray:
    return np.asarray(row["cycles_per_pixel_xy"], dtype=np.float64)


def distance_bins(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) * 512)


def main() -> None:
    matched = json.loads(MATCHED.read_text(encoding="utf-8"))
    d5, d6 = matched["scans"]
    if d5["printer_id"] != "D5_HPCLJ5550" or d6["printer_id"] != "D6_HPCLJ5550":
        raise ValueError("training/holdout printer IDs changed")
    candidates = d5["observations"]["blue"]["top_five_per_rgb"]["G"][:2]
    if not len(candidates) == 2:
        raise ValueError("D5 does not have two lattice candidates")
    # Choose the candidate in the lower-right half-plane to define a basis;
    # the observed second direction is compared with its orthogonal prediction.
    basis = next(xy(row) for row in candidates if xy(row)[0] > 0)
    second = next(xy(row) for row in candidates if xy(row)[0] < 0)
    effective_lpi = float(800 * (np.linalg.norm(basis) + np.linalg.norm(second)) / 2)
    angle = float(np.degrees(np.arctan2(basis[1], basis[0])))
    f = effective_lpi / 800
    theory_a = np.array([f * np.cos(np.deg2rad(angle)), f * np.sin(np.deg2rad(angle))])
    theory_b = np.array([f * np.sin(np.deg2rad(angle)), -f * np.cos(np.deg2rad(angle))])
    # A single partial-cyan separation creates one square dot lattice. The
    # source color is a structural probe, not DFD's unreleased digital tile.
    source = np.empty((512, 512, 3), dtype=np.float32)
    source[:] = (0.8, 1.0, 1.0)
    parameters = ColorPrintScanParameters(
        digital_ppi=800,
        render_ppi=2400,
        scan_ppi=800,
        screen_lpi=effective_lpi,
        screen_angles_degrees=(angle, 75.0, 0.0, 45.0),
        scanner_gamma=1.0,
        max_render_pixels=3_000_000,
    )
    simulation = simulate_color_print_scan(source, parameters)
    simulated_peaks = top_spectral_peaks(simulation.scanner_output_rgb, 0, limit=8)
    d6_peaks = d6["observations"]["blue"]["top_five_per_rgb"]["G"][:2]
    heldout = [xy(row) for row in d6_peaks]
    report = {
        "calibration": "D5 blue chart crop; top two G scan peaks; choose positive-x vector as basis; infer effective square-grid lpi and angle",
        "holdout": "D6 same-model same-named setting, untouched during parameter choice",
        "not_identified": "RIP screen fundamentals, CMYK ink assignment, phase, dot gain, paper spectrum, scanner MTF",
        "source_hypothesis": "uniform encoded RGB (.8,1,1), partial cyan only, 512x512 at 800 digital ppi",
        "effective_screen_lpi_assuming_fundamental": effective_lpi,
        "effective_screen_angle_degrees_in_scan_axes": angle,
        "d5_calibration_vectors": [basis.tolist(), second.tolist()],
        "forward_theory_vectors": [theory_a.tolist(), theory_b.tolist()],
        "d6_observed_top_two_g_vectors": [row.tolist() for row in heldout],
        "theory_to_d6_nearest_distance_fft_bins": [
            min(distance_bins(theory, observed) for observed in heldout)
            for theory in (theory_a, theory_b)
        ],
        "simulated_red_channel_top_eight": simulated_peaks,
        "theory_to_sim_nearest_distance_fft_bins": [
            min(distance_bins(theory, xy(row)) for row in simulated_peaks)
            for theory in (theory_a, theory_b)
        ],
    }
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "effective_screen_lpi_assuming_fundamental",
                    "effective_screen_angle_degrees_in_scan_axes",
                    "theory_to_d6_nearest_distance_fft_bins",
                    "theory_to_sim_nearest_distance_fft_bins",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
