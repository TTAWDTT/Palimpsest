"""One fixed-condition sanity comparison, not a calibrated printer validation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from origin_simulation.print_scan import PrintScanParameters, simulate_print_scan


SOURCE = Path(r"E:\ai_image_origin_research\data\derived\dfd_probe\D5_DC1_x_800_P3_S1_T1_2111_1.tiff")
ROI = (5100, 1800, 5612, 2312)


def features(patch: np.ndarray) -> dict:
    image = np.asarray(patch, dtype=np.float64)
    window = np.hanning(512)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((image - image.mean()) * window[:, None] * window[None, :])))
    yy, xx = np.indices((512, 512))
    radius = np.hypot(xx - 256, yy - 256)
    annulus = (radius > 65) & (radius < 110)
    peak_region = (abs(xx - 192) <= 1) & (abs(yy - 192) <= 1)
    return {
        "mean": float(image.mean()), "std": float(image.std()),
        "at_observed_peak_over_annulus_median": float(spectrum[peak_region].max() / np.median(spectrum[annulus])),
        "at_observed_peak_amplitude": float(spectrum[peak_region].max()),
    }


with Image.open(SOURCE) as image:
    real = np.asarray(image.crop(ROI), dtype=np.float32) / 255

# Frequency is taken directly from the D5 observed peak, not from an RIP job.
# The digital tone and reflectances below are illustrative guesses, not fitted
# source-chart values. A same-frequency model can still be physically wrong.
params = PrintScanParameters(
    digital_ppi=800, render_ppi=2400, scan_ppi=800,
    screen_lpi=np.sqrt(2) * 100, screen_angle_degrees=45,
    paper_reflectance=.932, ink_reflectance=.10,
    paper_scatter_sigma_um=10, scanner_optical_sigma_um=10,
)
sim = simulate_print_scan(np.full((512, 512), .73, dtype=np.float32), params)
noise_proxy = simulate_print_scan(
    np.full((512, 512), .73, dtype=np.float32),
    replace(params, scanner_noise_std_linear=.028),
)
blur_diagnostic = []
for sigma_um in (10, 20, 30, 40, 55):
    variant = replace(params, scanner_optical_sigma_um=sigma_um,
                      scanner_noise_std_linear=.028)
    blur_diagnostic.append({"scanner_optical_sigma_um": sigma_um,
                            **features(simulate_print_scan(
                                np.full((512, 512), .73, dtype=np.float32), variant
                            ).scanner_output)})
out = {
    "scope": "one real uniform gray tile vs virtual constant-tone square-spot model; observed peak used as model input",
    "real": features(real),
    "virtual": features(sim.scanner_output),
    "virtual_with_blank_paper_std_as_noise_proxy": features(noise_proxy.scanner_output),
    "blur_diagnostic_same_real_tile_no_holdout": blur_diagnostic,
    "virtual_parameters": {k: float(v) if isinstance(v, np.floating) else v for k, v in vars(params).items()},
}
Path("work/dfd_print_scan_prototype_comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"real": out["real"], "virtual": out["virtual"],
                  "virtual_with_noise_proxy": out["virtual_with_blank_paper_std_as_noise_proxy"],
                  "blur_diagnostic": blur_diagnostic}, indent=2))
