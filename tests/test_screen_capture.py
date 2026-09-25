import numpy as np

from origin_simulation.screen_capture import (
    ScreenCaptureParameters, _airy_kernel, _bayer_masks, render_screen_capture,
)


def parameters(display_pixels_per_sensor_pixel: float, blur_sigma: float = 0.0):
    return ScreenCaptureParameters(
        sensor_to_display=np.array(
            [[display_pixels_per_sensor_pixel, 0.0, 5.0],
             [0.0, display_pixels_per_sensor_pixel, 5.0],
             [0.0, 0.0, 1.0]]
        ),
        optical_blur_sigma_sensor_pixels=blur_sigma,
    )


def test_configurable_bayer_phase_changes_raw_sampling_sites():
    expected = {
        "RGGB": [[0, 1], [1, 2]],
        "BGGR": [[2, 1], [1, 0]],
        "GRBG": [[1, 0], [2, 1]],
        "GBRG": [[1, 2], [0, 1]],
    }
    for name, layout in expected.items():
        actual = np.argmax(_bayer_masks((2, 2), name), axis=-1)
        np.testing.assert_array_equal(actual, layout)

    red = np.zeros((20, 20, 3), dtype=np.float32)
    red[..., 0] = 1
    outputs = {}
    for name in ("RGGB", "BGGR"):
        setup = ScreenCaptureParameters(sensor_to_display=np.eye(3), fill_fraction=1,
                                        emitter_layout="co_spatial_rgb_control",
                                        display_gamma=1, sensor_bayer_pattern=name)
        outputs[name] = render_screen_capture(red, (8, 8), setup).noiseless_mosaic
    assert outputs["RGGB"][0, 0] > 0
    assert outputs["RGGB"][1, 1] == 0
    assert outputs["BGGR"][0, 0] == 0
    assert outputs["BGGR"][1, 1] > 0
    with np.testing.assert_raises_regex(ValueError, "Bayer pattern"):
        ScreenCaptureParameters(sensor_to_display=np.eye(3), sensor_bayer_pattern="RGB")


def dominant_horizontal_frequency(irradiance: np.ndarray) -> float:
    line = irradiance[..., 1].mean(axis=0)
    spectrum = np.abs(np.fft.rfft(line - line.mean()))
    spectrum[0] = 0.0
    return np.fft.rfftfreq(line.size)[np.argmax(spectrum)]


def test_projected_pitch_moves_screen_grid_frequency():
    display = np.ones((100, 100, 3), dtype=np.float32)
    near = render_screen_capture(display, (256, 256), parameters(0.25), samples_per_sensor_pixel=8)
    far = render_screen_capture(display, (256, 256), parameters(0.125), samples_per_sensor_pixel=8)

    assert abs(dominant_horizontal_frequency(near.irradiance) - 0.25) < 0.02
    assert abs(dominant_horizontal_frequency(far.irradiance) - 0.125) < 0.02


def test_defocus_suppresses_grid_without_moving_its_frequency():
    display = np.ones((100, 100, 3), dtype=np.float32)
    sharp = render_screen_capture(display, (256, 256), parameters(0.25), samples_per_sensor_pixel=8)
    blurred = render_screen_capture(display, (256, 256), parameters(0.25, 1.1), samples_per_sensor_pixel=8)

    sharp_line = sharp.irradiance[..., 1].mean(axis=0)
    blurred_line = blurred.irradiance[..., 1].mean(axis=0)
    assert abs(dominant_horizontal_frequency(sharp.irradiance) -
               dominant_horizontal_frequency(blurred.irradiance)) < 0.01
    assert blurred_line.std() < sharp_line.std() * 0.65


def test_circular_aperture_diffraction_psf_and_grid_response():
    narrow = _airy_kernel(30, 5.0)
    wide = _airy_kernel(60, 10.0)
    assert np.isclose(narrow.sum(), 1, atol=1e-6)
    assert np.isclose(wide.sum(), 1, atol=1e-6)
    assert wide[60, 60] < narrow[30, 30]

    display = np.ones((100, 100, 3), dtype=np.float32)
    sharp = render_screen_capture(display, (96, 96), parameters(0.25),
                                  samples_per_sensor_pixel=8)
    setup = ScreenCaptureParameters(
        sensor_to_display=parameters(0.25).sensor_to_display,
        diffraction_f_number=11,
        sensor_pixel_pitch_um=4.30652,
    )
    diffracted = render_screen_capture(display, (96, 96), setup,
                                       samples_per_sensor_pixel=8)
    assert np.isfinite(diffracted.irradiance).all()
    assert diffracted.irradiance.min() >= 0
    assert abs(dominant_horizontal_frequency(sharp.irradiance) -
               dominant_horizontal_frequency(diffracted.irradiance)) < 0.02
    assert diffracted.irradiance[..., 1].mean(axis=0).std() < \
        sharp.irradiance[..., 1].mean(axis=0).std() * 0.8


def test_diffraction_needs_physical_scale():
    with np.testing.assert_raises_regex(ValueError, "pixel pitch"):
        ScreenCaptureParameters(sensor_to_display=np.eye(3), diffraction_f_number=11)


def test_sensor_noise_is_seeded_and_signal_dependent():
    display = np.zeros((100, 100, 3), dtype=np.float32)
    display[:, :50] = 0.2
    display[:, 50:] = 0.8
    setup = ScreenCaptureParameters(
        sensor_to_display=np.array([[0.8, 0, 5], [0, 0.8, 5], [0, 0, 1]], dtype=float),
        fill_fraction=1.0,
        exposure_electrons_per_unit=2000,
        full_well_electrons=10000,
        read_noise_electrons=0.0,
    )
    first = render_screen_capture(display, (80, 80), setup, seed=42, samples_per_sensor_pixel=24)
    second = render_screen_capture(display, (80, 80), setup, seed=42, samples_per_sensor_pixel=24)

    assert np.array_equal(first.raw_mosaic, second.raw_mosaic)
    residual = first.raw_mosaic - first.noiseless_mosaic
    assert residual[:, 60:75].var() > residual[:, 10:40].var()


def test_output_and_intermediates_have_expected_ranges():
    display = np.full((64, 64, 3), 0.5, dtype=np.float32)
    result = render_screen_capture(display, (32, 48), parameters(0.5), samples_per_sensor_pixel=16)
    assert result.srgb.shape == (32, 48, 3)
    assert result.irradiance.shape == (32, 48, 3)
    assert result.raw_mosaic.shape == (32, 48)
    assert np.isfinite(result.srgb).all()
    assert 0 <= result.srgb.min() <= result.srgb.max() <= 1


def test_uniform_screen_integration_converges_across_phase_and_grid():
    display = np.ones((100, 100, 3), dtype=np.float32)
    expected = 0.85**2 / 3
    means = []
    for phase in (8.0, 8.1):
        transform = np.array([[2, 0, phase], [0, 2, phase], [0, 0, 1]], dtype=float)
        setup = ScreenCaptureParameters(sensor_to_display=transform)
        fine = render_screen_capture(display, (16, 16), setup, samples_per_sensor_pixel=64)
        coarse = render_screen_capture(display, (16, 16), setup, samples_per_sensor_pixel=60)
        means.append(fine.irradiance.mean(axis=(0, 1)))
        assert np.max(np.abs(fine.irradiance.mean(axis=(0, 1)) -
                             coarse.irradiance.mean(axis=(0, 1)))) < 0.02
    assert np.max(np.abs(np.asarray(means) - expected)) < 0.02


def test_rejects_underresolved_emitter_and_nonfinite_parameters():
    display = np.ones((32, 32, 3), dtype=np.float32)
    setup = ScreenCaptureParameters(
        sensor_to_display=np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=float)
    )
    with np.testing.assert_raises_regex(ValueError, "underresolved"):
        render_screen_capture(display, (8, 8), setup, samples_per_sensor_pixel=4)
    with np.testing.assert_raises_regex(ValueError, "integer"):
        render_screen_capture(display, (8, 8), setup, samples_per_sensor_pixel=4.0)
    with np.testing.assert_raises(ValueError):
        ScreenCaptureParameters(sensor_to_display=np.eye(3), display_gamma=np.nan)


def test_exposure_changes_signal_without_changing_full_well():
    display = np.ones((32, 32, 3), dtype=np.float32)
    transform = np.array([[0.25, 0, 5], [0, 0.25, 5], [0, 0, 1]], dtype=float)
    low = ScreenCaptureParameters(
        sensor_to_display=transform, fill_fraction=1.0,
        exposure_electrons_per_unit=2000, full_well_electrons=10000,
    )
    high = ScreenCaptureParameters(
        sensor_to_display=transform, fill_fraction=1.0,
        exposure_electrons_per_unit=4000, full_well_electrons=10000,
    )
    first = render_screen_capture(display, (32, 32), low)
    second = render_screen_capture(display, (32, 32), high)
    assert np.isclose(second.noiseless_mosaic.mean(), 2 * first.noiseless_mosaic.mean())
    assert not low.sensor_to_display.flags.writeable


def test_zero_exposure_has_zero_expected_photons_and_read_noise():
    display = np.ones((32, 32, 3), dtype=np.float32)
    transform = np.array([[0.25, 0, 5], [0, 0.25, 5], [0, 0, 1]], dtype=float)
    darkness = ScreenCaptureParameters(
        sensor_to_display=transform, exposure_electrons_per_unit=0,
        full_well_electrons=10000, read_noise_electrons=100,
    )
    dim = ScreenCaptureParameters(
        sensor_to_display=transform, exposure_electrons_per_unit=100,
        full_well_electrons=10000, read_noise_electrons=100,
    )
    zero = render_screen_capture(display, (32, 32), darkness)
    positive = render_screen_capture(display, (32, 32), dim)
    assert np.all(zero.noiseless_mosaic == 0)
    assert positive.noiseless_mosaic.mean() > zero.noiseless_mosaic.mean()
    assert zero.raw_mosaic.mean() > 0
