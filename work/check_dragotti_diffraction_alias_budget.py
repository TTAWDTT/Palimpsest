"""Conditional diffraction/alias budget for Dragotti EOS600D recapture.

No device parameters are inferred from the PNG. Pixel pitch and f-numbers
are source-reported nominal/example values, and display pitch is conditional
on source filling 1080 screen rows with no later output resize. The model is
an ideal monochromatic incoherent circular pupil, not a measured camera MTF.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


AUDIT = Path("outputs/03_过程模拟/Dragotti跨设备配对审计_2026-09-24.json")
CROSS = Path("outputs/03_过程模拟/Dragotti跨场景几何复核_2026-09-24.json")
OUT = Path(__file__).with_name("dragotti_diffraction_alias_budget.json")
PIXEL_PITCH_UM = 4.30652
WAVELENGTH_NM = {"blue_proxy": 460, "green_proxy": 540, "red_proxy": 610}
F_NUMBERS = (8, 11, 13)


def ideal_circular_pupil_mtf(frequency: float, cutoff: float) -> float:
    rho = frequency / cutoff
    if rho >= 1:
        return 0.0
    return 2 / math.pi * (math.acos(rho) - rho * math.sqrt(1 - rho * rho))


def main() -> None:
    record = json.loads(AUDIT.read_text(encoding="utf-8"))
    samples = [s for s in record["samples"] if s["source_key"] == "D40-015"]
    if len(samples) != 1:
        raise RuntimeError("source D40-015 audit not unique")
    eos = [s for s in samples[0]["recaptured"] if s["camera"] == "EOS600D"]
    if len(eos) != 1:
        raise RuntimeError("EOS600D sample not unique")
    pitch = eos[0]["conditional_projected_screen_pitch_if_fit_1080_rows"]
    frequency = 1 / pitch
    rows = []
    for f_number in F_NUMBERS:
        for name, wavelength_nm in WAVELENGTH_NM.items():
            wavelength_um = wavelength_nm / 1000
            cutoff = PIXEL_PITCH_UM / (wavelength_um * f_number)
            rows.append({"f_number": f_number, "wavelength_proxy": name,
                         "wavelength_nm": wavelength_nm,
                         "ideal_incoherent_cutoff_cycles_per_sensor_pixel": cutoff,
                         "screen_frequency_over_cutoff": frequency / cutoff,
                         "ideal_monochromatic_mtf_at_screen_frequency":
                             ideal_circular_pupil_mtf(frequency, cutoff),
                         "critical_f_number_for_zero_fundamental":
                             PIXEL_PITCH_UM * pitch / wavelength_um})
    output = {"audit_source": str(AUDIT),
              "conditional_screen_pitch_sensor_pixels": pitch,
              "screen_lattice_fundamental_cycles_per_sensor_pixel": frequency,
              "unfiltered_one_dimensional_alias_cycles_per_sensor_pixel": abs(1 - frequency),
              "unfiltered_one_dimensional_alias_period_sensor_pixels": 1 / abs(1 - frequency),
              "eos600d_pixel_pitch_um_from_author_example": PIXEL_PITCH_UM,
              "f8_status": "hypothetical intervention value, not claimed for this dataset",
              "f11_status": "2015 paper/2014 thesis textual design example, not per-image EXIF",
              "f13_status": "2014 thesis camera-level table, not per-image EXIF",
              "wavelengths": "illustrative monochromatic band proxies, not measured display spectra",
              "formula": "ideal incoherent cutoff = sensor pitch/(wavelength*f-number) cycles/sensor pixel; circular-pupil MTF=2/pi*(acos(rho)-rho*sqrt(1-rho^2))",
              "assumptions": [
                  "source image fills 1080 display rows with preserved aspect ratio",
                  "published recapture was cropped but not resized",
                  "projected screen-grid period equals conditional pitch",
                  "nominal f-number is treated as image-side effective f-number",
                  "monochromatic incoherent diffraction-limited circular pupil",
              ],
              "cases": rows,
              "cross_scene_json_available": CROSS.is_file()}
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pitch": pitch, "screen_frequency": frequency,
                      "unfiltered_alias_period": output["unfiltered_one_dimensional_alias_period_sensor_pixels"],
                      "cases": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
