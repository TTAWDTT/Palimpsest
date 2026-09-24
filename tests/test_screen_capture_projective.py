import numpy as np
from scipy.ndimage import gaussian_filter

from origin_simulation.screen_capture import ScreenCaptureParameters, _optical_blur, render_screen_capture


def test_projective_emitter_quadrature_converges_on_uniform_screen():
    display = np.ones((80, 80, 3), dtype=np.float32)
    homography = np.array([[.7, .03, 8], [.02, .65, 8], [.002, .001, 1]])
    setup = ScreenCaptureParameters(sensor_to_display=homography)
    medium = render_screen_capture(display, (12, 12), setup, samples_per_sensor_pixel=32)
    fine = render_screen_capture(display, (12, 12), setup, samples_per_sensor_pixel=48)
    assert np.isfinite(fine.irradiance).all()
    assert np.max(np.abs(medium.irradiance - fine.irradiance)) < .025
    assert abs(fine.irradiance.mean() - .85**2 / 3) < .02


def test_projective_grid_responds_to_homography_tilt():
    display = np.ones((100, 100, 3), dtype=np.float32)
    front = ScreenCaptureParameters(sensor_to_display=np.array([[.25, 0, 5], [0, .25, 5], [0, 0, 1]]))
    tilted = ScreenCaptureParameters(sensor_to_display=np.array([[.25, .03, 5], [.01, .25, 5],
                                                                  [.002, 0, 1]]))
    first = render_screen_capture(display, (24, 24), front, samples_per_sensor_pixel=16)
    second = render_screen_capture(display, (24, 24), tilted, samples_per_sensor_pixel=32)
    assert second.irradiance.shape == first.irradiance.shape
    assert not np.allclose(first.irradiance, second.irradiance)
    with np.testing.assert_raises_regex(ValueError, "horizon"):
        invalid = ScreenCaptureParameters(sensor_to_display=np.array([[1, 0, 0], [0, 1, 0],
                                                                       [-.1, 0, 1]]))
        render_screen_capture(display, (24, 24), invalid)


def test_rolling_exposure_and_pwm_generate_rows_only_when_temporally_aliasing():
    display = np.ones((40, 40, 3), dtype=np.float32)
    homography = np.array([[.25, 0, 5], [0, .25, 5], [0, 0, 1]])
    short = ScreenCaptureParameters(sensor_to_display=homography, fill_fraction=1,
                                    pwm_frequency_hz=100, pwm_duty_cycle=.25,
                                    exposure_time_s=.002, sensor_row_interval_s=.002)
    long = ScreenCaptureParameters(sensor_to_display=homography, fill_fraction=1,
                                   pwm_frequency_hz=100, pwm_duty_cycle=.25,
                                   exposure_time_s=.01, sensor_row_interval_s=.002)
    moving = render_screen_capture(display, (12, 12), short)
    averaged = render_screen_capture(display, (12, 12), long)
    assert np.allclose(moving.row_exposure_gain[:5], [1, .25, 0, 0, 0])
    assert np.allclose(averaged.row_exposure_gain, .25)
    assert np.allclose(moving.irradiance.mean(axis=(1, 2))[:5], moving.row_exposure_gain[:5] / 3)
    with np.testing.assert_raises_regex(ValueError, "PWM requires"):
        ScreenCaptureParameters(sensor_to_display=homography, pwm_frequency_hz=100)


def test_tiled_render_matches_whole_crop_with_optical_halo():
    y, x = np.mgrid[0:64, 0:64]
    display = np.stack(((x % 7) / 6, (y % 9) / 8, ((x + y) % 11) / 10), axis=-1).astype(np.float32)
    transforms = (
        np.array([[.5, 0, 8], [0, .5, 8], [0, 0, 1]]),
        np.array([[.55, .03, 8], [.02, .52, 8], [.001, .0005, 1]]),
    )
    for homography in transforms:
        setup = ScreenCaptureParameters(sensor_to_display=homography,
                                        optical_blur_sigma_sensor_pixels=.7)
        whole = render_screen_capture(display, (24, 20), setup, samples_per_sensor_pixel=32)
        tiled = render_screen_capture(display, (24, 20), setup,
                                      samples_per_sensor_pixel=32, tile_size_sensor_pixels=8)
        assert np.max(np.abs(whole.irradiance - tiled.irradiance)) < 2e-5
        assert np.allclose(whole.raw_mosaic, tiled.raw_mosaic, atol=2e-5, rtol=0)


def test_fft_optical_blur_matches_reflected_spatial_filter():
    fine = np.random.default_rng(9).random((480, 480, 3), dtype=np.float32)
    direct = gaussian_filter(fine, (8, 8, 0), mode="reflect")
    fast = _optical_blur(fine, 8)
    assert np.max(np.abs(direct - fast)) < 2e-6


def test_shutter_duration_couples_photon_count_to_pwm_integration():
    display = np.ones((40, 40, 3), dtype=np.float32)
    homography = np.array([[.25, 0, 5], [0, .25, 5], [0, 0, 1]])
    common = dict(sensor_to_display=homography, fill_fraction=1,
                  electron_rate_per_unit_s=200_000, full_well_electrons=10000,
                  pwm_frequency_hz=100, pwm_duty_cycle=.5, sensor_row_interval_s=.001)
    short = render_screen_capture(display, (20, 20),
                                  ScreenCaptureParameters(**common, exposure_time_s=.01))
    long = render_screen_capture(display, (20, 20),
                                 ScreenCaptureParameters(**common, exposure_time_s=.02))
    assert np.allclose(short.row_exposure_gain, .5)
    assert np.allclose(long.row_exposure_gain, .5)
    assert np.isclose(long.noiseless_mosaic.mean(), 2 * short.noiseless_mosaic.mean())
    with np.testing.assert_raises_regex(ValueError, "either integrated electrons or electron rate"):
        ScreenCaptureParameters(**common, exposure_time_s=.01, exposure_electrons_per_unit=1000)


def test_pwm_duty_reduces_photons_for_fixed_peak_radiance():
    display = np.ones((40, 40, 3), dtype=np.float32)
    homography = np.array([[.25, 0, 5], [0, .25, 5], [0, 0, 1]])
    common = dict(sensor_to_display=homography, fill_fraction=1,
                  pwm_frequency_hz=100, exposure_time_s=.01,
                  electron_rate_per_unit_s=100_000, full_well_electrons=10_000)
    always = render_screen_capture(display, (12, 12),
                                   ScreenCaptureParameters(**common, pwm_duty_cycle=1))
    quarter = render_screen_capture(display, (12, 12),
                                    ScreenCaptureParameters(**common, pwm_duty_cycle=.25))
    assert np.allclose(always.row_exposure_gain, 1)
    assert np.allclose(quarter.row_exposure_gain, .25)
    assert np.isclose(quarter.noiseless_mosaic.mean(), always.noiseless_mosaic.mean() / 4)


def test_display_lattice_alias_frequency_tracks_projected_pixel_scale():
    # A white digital image still emits through a physical subpixel lattice.
    # Its observed frequency must fold at the sensor Nyquist limit as the
    # camera-to-screen scale changes; this is a causal check, not image matching.
    display = np.ones((128, 512, 3), dtype=np.float32)
    for scale in (.65, .85, 1.15):
        H = np.array([[scale, 0, 50.33], [0, .18, 50.25], [0, 0, 1.]])
        setup = ScreenCaptureParameters(sensor_to_display=H)
        oversampling = max(32, int(np.ceil(24 * scale / setup.fill_fraction)))
        result = render_screen_capture(display, (4, 128), setup,
                                       samples_per_sensor_pixel=oversampling,
                                       tile_size_sensor_pixels=32)
        signal = result.irradiance[:, :, 1].mean(axis=0)
        spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
        spectrum[0] = 0
        measured = np.argmax(spectrum) / 128
        expected = abs(scale - round(scale))
        assert abs(measured - expected) <= 1 / 128


def test_presampling_optical_blur_suppresses_lattice_alias():
    display = np.ones((128, 512, 3), dtype=np.float32)
    H = np.array([[.85, 0, 50.33], [0, .18, 50.25], [0, 0, 1.]])
    amplitudes = []
    for blur_sigma in (0, .5, 1.0):
        setup = ScreenCaptureParameters(sensor_to_display=H,
                                        optical_blur_sigma_sensor_pixels=blur_sigma)
        rendered = render_screen_capture(display, (4, 256), setup,
                                         samples_per_sensor_pixel=32,
                                         tile_size_sensor_pixels=32)
        # Ignore global crop boundaries: a finite reflected PSF can create
        # artificial edge energy unrelated to the interior screen lattice.
        signal = rendered.irradiance[:, :, 1].mean(axis=0)[64:192]
        spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
        amplitudes.append(float(spectrum[round(.15 * 128)]))
    assert amplitudes[0] > 10 * amplitudes[1]
    assert amplitudes[1] > 100 * amplitudes[2]


def test_optional_post_tone_luma_sharpening_changes_rgb_not_raw():
    display = np.empty((40, 40, 3), dtype=np.float32)
    display[:] = (.15, .25, .35)
    display[:, 15:] = (.35, .45, .55)
    common = dict(sensor_to_display=np.array([[.5, 0, 5], [0, .5, 5], [0, 0, 1]]),
                  fill_fraction=1, display_gamma=1)
    baseline = render_screen_capture(display, (32, 32), ScreenCaptureParameters(**common),
                                     samples_per_sensor_pixel=16)
    enhanced = render_screen_capture(
        display, (32, 32),
        ScreenCaptureParameters(**common, isp_luma_sharpen_amount=.5,
                                isp_luma_sharpen_sigma_pixels=1),
        samples_per_sensor_pixel=16)
    assert np.array_equal(baseline.raw_mosaic, enhanced.raw_mosaic)
    assert np.array_equal(baseline.srgb, enhanced.srgb_before_sharpen)
    assert np.array_equal(baseline.srgb, baseline.srgb_before_sharpen)
    assert np.max(np.abs(enhanced.srgb - baseline.srgb)) > .001
    delta = enhanced.srgb - enhanced.srgb_before_sharpen
    unclipped = ((enhanced.srgb > .01) & (enhanced.srgb < .99)).all(axis=2)
    assert np.max(np.abs(delta[unclipped, 0] - delta[unclipped, 1])) < 1e-6
    assert np.max(np.abs(delta[unclipped, 1] - delta[unclipped, 2])) < 1e-6
    with np.testing.assert_raises_regex(ValueError, "sharpening"):
        ScreenCaptureParameters(**common, isp_luma_sharpen_amount=-.1)
