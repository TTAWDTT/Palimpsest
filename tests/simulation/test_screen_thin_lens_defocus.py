import numpy as np
import pytest

from palimpsest.simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)
from palimpsest.simulation._screen.optics import (
    thin_lens_coc_radius_sensor_pixels,
    thin_lens_frontoparallel_sensor_to_display,
)


def _params(
    *,
    focus_distance_m: float = 0.5,
    aperture_f_number: float = 2,
    sensor_to_display: np.ndarray | None = None,
) -> ScreenCaptureParameters:
    if sensor_to_display is None:
        sensor_to_display = np.asarray([[0.93, 0, 9.13], [0, 0.93, 8.22], [0, 0, 1]])
    return ScreenCaptureParameters(
        sensor_to_display=sensor_to_display,
        fill_fraction=0.75,
        emitter_layout="co_spatial_rgb_control",
        display_gamma=1,
        lens_focal_length_mm=4,
        aperture_f_number=aperture_f_number,
        screen_distance_m=0.5,
        focus_distance_m=focus_distance_m,
        sensor_pixel_pitch_um=4,
    )


def test_thin_lens_focus_distance_and_aperture_bind_defocus_geometry():
    focused = _params(focus_distance_m=0.5)
    wrong_focus = _params(focus_distance_m=1.0)
    stopped_down = _params(focus_distance_m=1.0, aperture_f_number=4)
    assert thin_lens_coc_radius_sensor_pixels(focused) == pytest.approx(0, abs=1e-12)
    assert thin_lens_coc_radius_sensor_pixels(wrong_focus) == pytest.approx(
        1.004016, rel=1e-5
    )
    assert thin_lens_coc_radius_sensor_pixels(stopped_down) == pytest.approx(
        thin_lens_coc_radius_sensor_pixels(wrong_focus) / 2, rel=1e-12
    )


def test_frontoparallel_projection_uses_actual_focal_plane_and_breathes():
    args = dict(
        lens_focal_length_mm=4,
        screen_distance_m=0.5,
        sensor_pixel_pitch_um=4,
        display_pixel_pitch_mm=0.25,
        display_origin_xy=(12.3, 8.4),
    )
    focused = thin_lens_frontoparallel_sensor_to_display(**args, focus_distance_m=0.5)
    farther = thin_lens_frontoparallel_sensor_to_display(**args, focus_distance_m=1)
    expected = 0.004 * 500 / (4 * 500 / (500 - 4) * 0.25)
    assert focused[0, 0] == pytest.approx(expected)
    assert focused[0, 2] == pytest.approx(12.3)
    assert farther[0, 0] > focused[0, 0]
    assert farther[1, 1] == farther[0, 0]
    with pytest.raises(ValueError, match="finite positive"):
        thin_lens_frontoparallel_sensor_to_display(
            **{**args, "display_origin_xy": (2,)}, focus_distance_m=0.5
        )


def test_defocus_disk_applies_before_sampling_and_preserves_flat_field_mean():
    display = np.ones((75, 75, 3), dtype=np.float32) * 0.6
    in_focus = render_screen_capture(
        display, (48, 48), _params(), spatial_method="fine"
    )
    defocused = render_screen_capture(
        display, (48, 48), _params(focus_distance_m=1), spatial_method="fine"
    )
    a = in_focus.irradiance[8:-8, 8:-8, 1]
    b = defocused.irradiance[8:-8, 8:-8, 1]
    assert abs(float(a.mean() - b.mean())) < 1e-3
    assert float(b.std()) < float(a.std()) * 0.5


def test_defocus_tile_halo_matches_single_tile_interior():
    frame = (
        np.random.default_rng(21).uniform(0.2, 0.8, size=(45, 45, 3)).astype(np.float32)
    )
    params = _params(focus_distance_m=1)
    whole = render_screen_capture(
        frame, (14, 14), params, spatial_method="fine", tile_size_sensor_pixels=14
    ).irradiance
    tiled = render_screen_capture(
        frame, (14, 14), params, spatial_method="fine", tile_size_sensor_pixels=7
    ).irradiance
    assert np.max(np.abs(whole - tiled)) < 2e-4


def test_screen_focus_limit_is_existing_airy_path():
    frame = (
        np.random.default_rng(8).uniform(0.1, 0.9, size=(36, 36, 3)).astype(np.float32)
    )
    focused = _params()
    diffraction_only = ScreenCaptureParameters(
        sensor_to_display=focused.sensor_to_display,
        fill_fraction=0.75,
        emitter_layout="co_spatial_rgb_control",
        display_gamma=1,
        diffraction_f_number=2,
        sensor_pixel_pitch_um=4,
    )
    a = render_screen_capture(frame, (9, 9), focused).irradiance
    b = render_screen_capture(frame, (9, 9), diffraction_only).irradiance
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_single_distance_approximation_rejects_incomplete_or_projective_geometry():
    with pytest.raises(ValueError, match="requires positive"):
        ScreenCaptureParameters(sensor_to_display=np.eye(3), lens_focal_length_mm=4)
    with pytest.raises(ValueError, match="distances must exceed"):
        _params(focus_distance_m=0.003)
    with pytest.raises(ValueError, match="must agree"):
        ScreenCaptureParameters(
            sensor_to_display=np.eye(3),
            lens_focal_length_mm=4,
            aperture_f_number=2,
            diffraction_f_number=4,
            screen_distance_m=0.5,
            focus_distance_m=1,
            sensor_pixel_pitch_um=4,
        )
    perspective = np.asarray([[0.93, 0, 9.13], [0, 0.93, 8.22], [0.001, 0, 1]])
    with pytest.raises(ValueError, match="affine projection"):
        _params(sensor_to_display=perspective)


def test_thin_lens_mode_requires_numerical_integrator():
    frame = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    with pytest.raises(ValueError, match="no diffraction"):
        render_screen_capture(frame, (8, 8), _params(), spatial_method="analytic")
    with pytest.raises(ValueError, match="Airy diffraction"):
        render_screen_capture(frame, (8, 8), _params(), spatial_method="prefilter")
