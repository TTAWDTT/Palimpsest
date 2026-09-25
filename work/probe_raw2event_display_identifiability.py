"""Explore whether virtual display pitch and blur are identifiable from known RAW.

The grid is deliberately fixed and small, with count response fit on the old
calibration ten only. This is exploratory; it does not identify actual LCD
resolution or PSF without the author's display input and capture metadata.
"""

import csv
import json
from pathlib import Path
import time

import numpy as np

from origin_simulation.screen_pipeline import DisplayRasterParameters, rasterize_display_source
from work.audit_raw2event_split_first_frames import load_originals
from work.evaluate_raw2event_global_geometry_process import fit_diagonal, evaluate
from work.probe_raw2event_source_to_raw import prepare
from work.probe_screen_display_prefilter import approximate_bands


SPLIT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
NEW = Path("E:/ai_image_origin_research/data/manifests/raw2event_phase_confirmation_v1.csv")
GEOM = Path("work/raw2event_sensor_geometry_audit.json")
OLD_CORNERS = Path("work/raw2event_content_registered_geometry.json")
NEW_CORNERS = Path("work/raw2event_phase_confirmation_evaluation.json")
OUT = Path("work/raw2event_display_identifiability_grid.json")
SIDES = (96, 192, 288)
SIGMAS = (0.55, 0.8, 1.1)


def main() -> None:
    rows = [r for r in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
            if r["role"] in ("calibration", "development")]
    rows += list(csv.DictReader(NEW.open(encoding="utf-8", newline="")))
    if len(rows) != 30 or len({r["prefix"] for r in rows}) != 30:
        raise RuntimeError("expected 30 already-used prefixes")
    originals = load_originals(rows)
    corners = {r["prefix"]: r["refined_corners"] for r in json.loads(OLD_CORNERS.read_text(encoding="utf-8"))["records"]}
    corners.update({r["prefix"]: r["geometry"]["source_rgb_refined_corners"]
                    for r in json.loads(NEW_CORNERS.read_text(encoding="utf-8"))["per_source"]})
    affine = np.asarray(json.loads(GEOM.read_text(encoding="utf-8"))["fit_matrices"]["global_affine"])
    prepared = {r["prefix"]: prepare(r["prefix"], None, originals[r["prefix"]],
                                      np.asarray(corners[r["prefix"]], np.float32), affine)
                for r in rows}
    runs = []
    for side in SIDES:
        drives = {r["prefix"]: rasterize_display_source(
                    originals[r["prefix"]], DisplayRasterParameters(
                        raster_size=(side, side), resampling="lanczos", resample_space="encoded_srgb"))
                  for r in rows}
        for sigma in SIGMAS:
            start = time.perf_counter()
            entries = []
            for number, row in enumerate(rows, 1):
                item = prepared[row["prefix"]]
                H = item["H"].copy()
                H[:2] *= side / 192
                bands = approximate_bands(drives[row["prefix"]], item["actual"].shape,
                                          H, .85, sigma, "vertical_rgb", 8, 4)
                entries.append((row, item, bands))
            result = evaluate(entries, fit_diagonal(entries))
            runs.append({"display_side": side, "sigma_sensor": sigma,
                         "elapsed_seconds": time.perf_counter() - start,
                         "weights": result["weights_intercept_gain_by_display_primary"],
                         "aggregate": result["aggregate"], "per_source": result["per_source"]})
            print(f"side={side} sigma={sigma} "
                  f"dev={result['aggregate']['development']['mean_source_mae_counts']:.3f} "
                  f"new={result['aggregate']['phase_confirmation']['mean_source_mae_counts']:.3f}", flush=True)
    report = {"scope": "3x3 virtual display-pitch/PSF grid on same 30 already-opened Raw2Event sources",
              "sides": SIDES, "sigmas": SIGMAS, "fit": "BGGR effective gains on ten calibration sources per condition",
              "render": "display prefilter S8/Q4; no actual panel calibration",
              "qualification": "exploratory, choices informed by prior data, roles not independent device replicates; cannot identify physical LCD pitch or PSF",
              "reserved_untouched": True, "runs": runs}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
