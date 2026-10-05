"""Check whether a flat-field spectral peak follows screen motion or sensor."""

from palimpsest.paths import WORK_DIR

import json

import numpy as np
from scipy.ndimage import gaussian_filter

from palimpsest.data.raw2event import ROOT, detect_tag, extract_frame


PREFIXES = (
    "252_airplane_1_3332_20251222_124809",
    "48340_ship_1_4036_20260117_195215",
)
FRAMES = (0, 79, 159, 238)
OUT = WORK_DIR / "raw2event_flat_field_phase_motion.json"


def complex_peak(raw: np.ndarray) -> dict:
    patch = raw[8:112, 32:660][0::2, 1::2].astype(np.float32)
    residual = patch - gaussian_filter(patch, sigma=(4, 5), mode="reflect")
    projection = residual.mean(axis=0) * np.hanning(patch.shape[1])
    fft = np.fft.rfft(projection)
    frequency = np.fft.rfftfreq(len(projection))
    index = int(np.argmin(abs(frequency - 0.25)))
    return {
        "frequency_cycles_per_cfa_plane_pixel": float(frequency[index]),
        "amplitude": float(abs(fft[index])),
        "phase_radians": float(np.angle(fft[index])),
        "neighbor_amplitude": [
            float(abs(fft[index + offset])) for offset in (-2, -1, 1, 2)
        ],
    }


def main() -> None:
    all_records = []
    for prefix in PREFIXES:
        frames = []
        for frame_index in FRAMES:
            raw = extract_frame(
                ROOT / "frames_raw" / f"{prefix}.mkv", frame_index, "gray16le", 1, "<u2"
            )
            rgb = extract_frame(
                ROOT / "frames_rgb" / f"{prefix}.mkv", frame_index, "rgb24", 3, "u1"
            )
            tag, tag_id = detect_tag(rgb)
            if tag_id != 0:
                raise RuntimeError(f"missing Tag in {prefix} frame {frame_index}")
            record = {
                "frame": frame_index,
                "rgb_tag_center_xy": tag.mean(axis=0).tolist(),
                "raw_flat_peak": complex_peak(raw),
            }
            frames.append(record)
            print(
                prefix,
                frame_index,
                record["rgb_tag_center_xy"],
                round(record["raw_flat_peak"]["phase_radians"], 3),
                flush=True,
            )
        all_records.append({"prefix": prefix, "frames": frames})
    report = {
        "scope": "fixed sensor-coordinate blank ROI, 0.25 cycles/CFA-plane-pixel",
        "status": "phase-vs-screen-motion diagnostic only; no pixel pitch inferred",
        "records": all_records,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
