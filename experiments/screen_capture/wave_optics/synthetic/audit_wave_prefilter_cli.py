"""Exercise the public single-image CLI for the bounded wave prefilter path."""

from palimpsest.paths import WORK_DIR

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    work = WORK_DIR.resolve()
    with tempfile.TemporaryDirectory(prefix="wave_prefilter_cli_", dir=work) as temp:
        root = Path(temp)
        source = root / "display.png"
        config = root / "config.json"
        prefix = root / "rendered"
        Image.fromarray(
            np.random.default_rng(37).integers(
                0, 256, size=(56, 56, 3), dtype=np.uint8
            ),
            mode="RGB",
        ).save(source)
        configuration = {
            "schema": "screen-forward-config-v1",
            "parameter_status": "virtual CLI smoke test, not device calibrated",
            "sensor_shape": [20, 20],
            "spatial_method": "wave_prefilter",
            "tile_size_sensor_pixels": 8,
            "parameters": {
                "sensor_to_display": [[0.93, 0, 9.137], [0, 0.93, 8.219], [0, 0, 1]],
                "fill_fraction": 0.85,
                "display_gamma": 1,
                "lens_focal_length_mm": 4,
                "aperture_f_number": 2,
                "screen_distance_m": 0.5,
                "focus_distance_m": 1,
                "sensor_pixel_pitch_um": 4,
                "defocus_psf_model": "wave",
            },
        }
        config.write_text(json.dumps(configuration), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "palimpsest.simulation.screen_capture.render_cli",
            "--input",
            str(source),
            "--config",
            str(config),
            "--output-prefix",
            str(prefix),
            "--intermediates",
        ]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        report = json.loads((root / "rendered.json").read_text(encoding="utf-8"))
        with Image.open(root / "rendered.png") as image:
            assert image.size == (20, 20)
        assert report["spatial_method"] == "wave_prefilter"
        assert (root / "rendered.npz").exists()
        print(
            json.dumps(
                {
                    "status": "passed",
                    "spatial_method": report["spatial_method"],
                    "seconds_render_only": report["seconds_render_only"],
                }
            )
        )


if __name__ == "__main__":
    main()
