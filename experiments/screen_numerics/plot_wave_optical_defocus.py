"""Visualize optical-PSF approximation error under one virtual setup."""

from palimpsest.paths import WORK_DIR

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = WORK_DIR / "wave_optical_defocus_psf_probe.json"
OUT = Path(
    "outputs/03_过程模拟/03_前向模型/薄透镜波动光学与圆盘近似频率响应_2026-09-25.png"
)


def main() -> None:
    rows = json.loads(DATA.read_text(encoding="utf-8"))["rows"]
    coc = np.asarray([row["geometric_coc_radius_sensor_pixels"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), constrained_layout=True)
    for axis, suffix, title in (
        (axes[0], "0p15", "Content: 0.15 cycles / sensor pixel"),
        (axes[1], "0p93", "Display lattice: 0.93 cycles / sensor pixel"),
    ):
        axis.plot(
            coc,
            [row[f"wave_transfer_{suffix}"] for row in rows],
            "o-",
            linewidth=2,
            label="circular-pupil wave PSF",
        )
        axis.plot(
            coc,
            [row[f"approx_transfer_{suffix}"] for row in rows],
            "s--",
            linewidth=2,
            label="Airy * geometric disk",
        )
        axis.set_title(title)
        axis.set_xlabel("Geometric CoC radius (sensor pixels)")
        axis.set_ylabel("Signed optical transfer")
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Virtual f=4 mm, f/2, screen=0.5 m, pitch=4 μm, λ=540 nm", fontsize=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=165)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
