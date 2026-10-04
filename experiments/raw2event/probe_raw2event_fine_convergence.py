"""Check whether the fast-vs-fine spectral gap comes from fine-grid aliasing."""

import csv
import json
from pathlib import Path

import numpy as np

from origin_simulation.screen_capture import (
    ScreenCaptureParameters,
    _minimum_samples_for_projection,
    render_screen_capture,
)
from experiments.raw2event.audit_raw2event_split_first_frames import load_originals
from experiments.raw2event.evaluate_raw2event_phase_confirmation import predict_phase
from experiments.raw2event.evaluate_raw2event_spectral_mix import (
    cached_bands,
    code_hash,
)
from experiments.raw2event.probe_raw2event_source_to_raw import prepare
from experiments.screen_numerics.probe_screen_display_prefilter import approximate_bands


MANIFEST = Path(
    "E:/ai_image_origin_research/data/manifests/raw2event_phase_confirmation_v1.csv"
)
GEOM = Path("work/raw2event_sensor_geometry_audit.json")
CORNERS = Path("work/raw2event_phase_confirmation_evaluation.json")
FIT = Path("work/raw2event_global_geometry_process_evaluation.json")
OUT = Path("work/raw2event_fine_convergence_probe.json")
CLASSES = ("airplane", "automobile", "truck")


def spectral_power(mosaic: np.ndarray) -> float:
    if mosaic.shape != (32, 32):
        raise ValueError("32x32 mosaic expected")
    window = np.outer(np.hanning(32), np.hanning(32))
    fft = np.fft.fft2((mosaic - mosaic.mean()) * window)
    freq = np.fft.fftfreq(32)
    radius = np.hypot(*np.meshgrid(freq, freq, indexing="ij"))
    return float((np.abs(fft) ** 2)[(radius >= 0.12) & (radius < 0.25)].sum())


def main() -> None:
    rows = [
        r
        for r in csv.DictReader(MANIFEST.open(encoding="utf-8", newline=""))
        if r["class_name"] in CLASSES
    ]
    if len(rows) != 3:
        raise RuntimeError("expected three fixed classes")
    original = load_originals(rows)
    corners = {
        r["prefix"]: r["geometry"]["source_rgb_refined_corners"]
        for r in json.loads(CORNERS.read_text(encoding="utf-8"))["per_source"]
    }
    affine = np.asarray(
        json.loads(GEOM.read_text(encoding="utf-8"))["fit_matrices"]["global_affine"]
    )
    weights = np.asarray(
        json.loads(FIT.read_text(encoding="utf-8"))["methods"]["vertical_rgb"][
            "weights_intercept_gain_by_display_primary"
        ]
    )
    results = []
    for row in rows:
        prefix = row["prefix"]
        prepared = prepare(
            prefix,
            None,
            original[prefix],
            np.asarray(corners[prefix], np.float32),
            affine,
        )
        fine, _ = cached_bands(row, prepared, "vertical_rgb", code_hash())
        fast = approximate_bands(
            prepared["drive"],
            prepared["actual"].shape,
            prepared["H"],
            0.85,
            0.8,
            "vertical_rgb",
            8,
            4,
        )
        yy, xx = np.nonzero(prepared["mask"])
        y0 = 2 * int((yy.mean() - 24) // 2)
        x0 = 2 * int((xx.mean() - 24) // 2)
        if x0 < 0 or y0 < 0 or x0 + 48 > fine.shape[1] or y0 + 48 > fine.shape[0]:
            raise RuntimeError("crop outside ROI")
        bounds = (slice(y0, y0 + 48), slice(x0, x0 + 48))
        H = prepared["H"] @ np.asarray([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
        params = ScreenCaptureParameters(
            sensor_to_display=H,
            fill_fraction=0.85,
            optical_blur_sigma_sensor_pixels=0.8,
        )
        requirement = _minimum_samples_for_projection(H, (48, 48), 0.85)
        hi = render_screen_capture(
            prepared["drive"],
            (48, 48),
            params,
            samples_per_sensor_pixel=64,
            tile_size_sensor_pixels=24,
        ).emitter_band_irradiance
        choices = {
            "fine_default": fine[bounds],
            "fast_prefilter": fast[bounds],
            "fine64": hi,
        }
        if row["class_name"] in ("automobile", "truck"):
            choices["fine128"] = render_screen_capture(
                prepared["drive"],
                (48, 48),
                params,
                samples_per_sensor_pixel=128,
                tile_size_sensor_pixels=16,
            ).emitter_band_irradiance
        mosaics = {
            name: predict_phase(bands, "BGGR", weights)[8:40, 8:40]
            for name, bands in choices.items()
        }
        spectral = {name: spectral_power(value) for name, value in mosaics.items()}
        reference_name = "fine128" if "fine128" in choices else "fine64"
        ref = choices[reference_name][8:40, 8:40]
        result = {
            "class_name": row["class_name"],
            "prefix": prefix,
            "roi_xy": [x0, y0],
            "crop_minimum_samples_per_axis": requirement,
            "reference": reference_name,
            "mean_abs_radiance_to_reference": {
                name: float(np.abs(bands[8:40, 8:40] - ref).mean())
                for name, bands in choices.items()
                if name != reference_name
            },
            "mid_band_power": spectral,
            "mid_band_delta_db_to_reference": {
                name: float(
                    10 * np.log10((power + 1e-12) / (spectral[reference_name] + 1e-12))
                )
                for name, power in spectral.items()
                if name != reference_name
            },
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    OUT.write_text(
        json.dumps(
            {
                "scope": "three preselected already-opened sources; 48x48 crop, inner32 spectrum",
                "fine_reference_qualification": "64 or 128 points per sensor axis, still numerical quadrature not exact truth",
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
