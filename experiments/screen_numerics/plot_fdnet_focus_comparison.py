"""Render a small, attributed visual comparison of verified author examples."""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATA = Path("E:/ai_image_origin_research/data/raw/fdnet_real_screen_pairs")
PROBE = Path("work/fdnet_focus_pair_process_probe.json")
OUT = Path("outputs/03_过程模拟/02_拍屏实测/FDNet真实失焦与成片模糊对照_2026-09-25.png")


def main() -> None:
    interventions = json.loads(PROBE.read_text(encoding="utf-8"))["interventions"]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), constrained_layout=True)
    for row, item in enumerate(interventions):
        name = item["name"]
        focused = np.asarray(Image.open(DATA / "moireresize" / name).convert("RGB"))
        defocused = np.asarray(Image.open(DATA / "blurresize" / name).convert("RGB"))
        sigma = item["postblur_matched_peak"]["sigma_output_pixels"]
        postblurred = cv2.GaussianBlur(focused, (0, 0), sigmaX=sigma)
        for axis, image, title in zip(
            axes[row],
            (focused, defocused, postblurred),
            (
                f"{name}: focused capture",
                "real defocused capture",
                f"post-capture Gaussian, σ={sigma:g} px",
            ),
        ):
            axis.imshow(image)
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Author-released RealScreenMoire examples; shown at released 256×256 resolution",
        fontsize=11,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
