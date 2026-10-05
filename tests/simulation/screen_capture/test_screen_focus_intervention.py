"""A screen lattice must be optically filtered before sensor sampling.

Once the lattice aliases to a low spatial frequency, blurring the sampled
image cannot remove it without also destroying genuine nearby content.
"""

import cv2
import numpy as np

from palimpsest.simulation.screen_capture.capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


def _amplitude(image: np.ndarray, frequency: float) -> float:
    line = image[16:-16, 16:-16, 1].mean(axis=0)
    line -= line.mean()
    window = np.hanning(len(line))
    return float(
        2
        * abs(
            np.sum(
                line * window * np.exp(-2j * np.pi * frequency * np.arange(len(line)))
            )
        )
        / window.sum()
    )


def _render(frame: np.ndarray, sigma: float) -> np.ndarray:
    setup = ScreenCaptureParameters(
        sensor_to_display=np.asarray(
            [[0.93, 0, 12.137], [0, 0.93, 11.219], [0, 0, 1.0]]
        ),
        fill_fraction=0.85,
        emitter_layout="co_spatial_rgb_control",
        display_gamma=1,
        optical_blur_sigma_sensor_pixels=sigma,
    )
    return render_screen_capture(
        frame, (128, 128), setup, spatial_method="analytic"
    ).irradiance


def test_presensor_focus_change_suppresses_grid_alias_while_preserving_content():
    flat = np.ones((200, 200, 3), dtype=np.float32) * 0.5
    x = np.arange(200)
    content = flat + 0.25 * np.sin(2 * np.pi * (0.15 / 0.93) * x)[None, :, None]
    focused_flat, defocused_flat = _render(flat, 0), _render(flat, 0.55)
    focused_content, defocused_content = _render(content, 0), _render(content, 0.55)
    alias_focus = _amplitude(focused_flat, 0.07)
    alias_defocus = _amplitude(defocused_flat, 0.07)
    content_focus = _amplitude(focused_content, 0.15)
    content_defocus = _amplitude(defocused_content, 0.15)
    postblur = cv2.GaussianBlur(focused_content, (0, 0), 7.25)
    postblur_content = _amplitude(postblur, 0.15)
    assert alias_focus > 0.002
    assert alias_defocus < alias_focus * 0.02
    assert content_defocus > content_focus * 0.80
    assert postblur_content < content_defocus * 0.01
