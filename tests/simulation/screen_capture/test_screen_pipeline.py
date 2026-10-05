"""Structure checks for digital display placement before physical recapture."""

import numpy as np

from palimpsest.simulation.digital.publication import PublicationParameters, apply_publication
from palimpsest.simulation.screen_capture.capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)
from palimpsest.simulation.screen_capture.pipeline import (
    DisplayRasterParameters,
    rasterize_display_source,
    run_screen_pipeline,
)


def test_display_placement_keeps_background_and_content_separate():
    source = np.full((2, 2, 3), [255, 0, 0], dtype=np.uint8)
    setup = DisplayRasterParameters(
        (8, 6), (2, 1, 6, 5), resampling="nearest", background_rgb=(0, 0, 32)
    )
    drive = rasterize_display_source(source, setup)
    assert drive.shape == (6, 8, 3)
    np.testing.assert_allclose(drive[0, 0], np.array([0, 0, 32]) / 255, atol=1e-7)
    np.testing.assert_array_equal(drive[2, 3], [1, 0, 0])


def test_linear_light_display_resize_differs_from_encoded_resize():
    source = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    encoded = rasterize_display_source(
        source,
        DisplayRasterParameters(
            (3, 1), resampling="bilinear", resample_space="encoded_srgb"
        ),
    )
    linear = rasterize_display_source(
        source,
        DisplayRasterParameters(
            (3, 1), resampling="bilinear", resample_space="linear_light"
        ),
    )
    assert 0.45 < encoded[0, 1, 0] < 0.55
    assert 0.70 < linear[0, 1, 0] < 0.77


def test_pipeline_matches_explicit_stage_order():
    source = np.random.default_rng(5).integers(0, 256, (12, 12, 3), dtype=np.uint8)
    display = DisplayRasterParameters((24, 24), resampling="bicubic")
    camera = ScreenCaptureParameters(
        sensor_to_display=np.array([[0.82, 0, 3], [0, 0.82, 3], [0, 0, 1]]),
        optical_blur_sigma_sensor_pixels=0.4,
        sensor_spectral_mix_rgb=((1, 0.1, 0), (0.2, 1, 0.1), (0, 0.1, 1)),
    )
    publication = PublicationParameters(
        crop_xyxy=(2, 2, 18, 18),
        output_size=(8, 8),
        encoding="jpeg",
        jpeg_quality=82,
        jpeg_subsampling=2,
    )
    actual = run_screen_pipeline(
        source, display, (20, 20), camera, publication, spatial_method="analytic"
    )
    manual_drive = rasterize_display_source(source, display)
    manual_capture = render_screen_capture(
        manual_drive, (20, 20), camera, spatial_method="analytic"
    )
    manual_publication = apply_publication(manual_capture.srgb, publication)
    np.testing.assert_array_equal(actual.display_drive_rgb, manual_drive)
    np.testing.assert_array_equal(actual.capture.raw_mosaic, manual_capture.raw_mosaic)
    np.testing.assert_array_equal(
        actual.publication.decoded_rgb, manual_publication.decoded_rgb
    )
    assert actual.publication.decoded_rgb.shape == (8, 8, 3)


def test_invalid_display_content_bounds_rejected():
    with np.testing.assert_raises_regex(ValueError, "rectangle"):
        DisplayRasterParameters((16, 16), content_xyxy=(0, 0, 17, 16))
