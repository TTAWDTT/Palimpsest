"""Validated screen, lens, exposure and sensor parameters; result arrays."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScreenCaptureParameters:
    """Explicit forward-model assumptions, not automatically calibrated values.

    Coordinates map sensor pixel centers to display pixel coordinates. Spatial
    blur is measured in sensor pixels; distances, wavelengths and exposure use
    the units in each field name. ``exposure_electrons_per_unit=None`` with no
    electron rate selects a noiseless preview. Thin-lens fields must be supplied
    together. Keep field order stable for existing positional construction.
    """

    # Geometry and display response.
    sensor_to_display: np.ndarray
    fill_fraction: float = 0.85
    emitter_layout: str = "vertical_rgb"
    display_gamma: float = 2.2
    # Optical transfer and optional physical thin-lens assumptions.
    optical_blur_sigma_sensor_pixels: float = 0.0
    diffraction_f_number: float | None = None
    sensor_pixel_pitch_um: float | None = None
    lens_focal_length_mm: float | None = None
    aperture_f_number: float | None = None
    screen_distance_m: float | None = None
    focus_distance_m: float | None = None
    defocus_psf_model: str = "geometric_airy"
    rgb_effective_wavelengths_nm: tuple[float, float, float] = (610.0, 540.0, 460.0)
    # Effective display-primary response of the sensor channels.
    sensor_spectral_mix_rgb: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    sensor_bayer_pattern: str = "RGGB"
    # Photon collection and readout. Integrated exposure and rate are exclusive.
    exposure_electrons_per_unit: float | None = None
    electron_rate_per_unit_s: float | None = None
    full_well_electrons: float = 10000.0
    read_noise_electrons: float = 0.0
    sensor_analog_gain_relative: float = 1.0
    # Global display PWM integrated over rolling sensor-row exposures.
    pwm_frequency_hz: float | None = None
    pwm_duty_cycle: float = 1.0
    pwm_off_level: float = 0.0
    exposure_time_s: float | None = None
    sensor_row_interval_s: float = 0.0
    pwm_phase_cycles: float = 0.0
    # Optional post-tone sharpening and relative aperture throughput.
    isp_luma_sharpen_amount: float = 0.0
    isp_luma_sharpen_sigma_pixels: float = 1.0
    throughput_reference_f_number: float | None = None

    def __post_init__(self) -> None:
        transform = np.asarray(self.sensor_to_display, dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("sensor_to_display must be a finite 3x3 matrix")
        normalizer = (
            transform[2, 2]
            if abs(transform[2, 2]) > 1e-12 * np.linalg.norm(transform)
            else np.linalg.norm(transform)
        )
        transform = transform / normalizer
        if abs(np.linalg.det(transform)) < 1e-12:
            raise ValueError("sensor_to_display must be invertible")
        scalars = (
            self.fill_fraction,
            self.display_gamma,
            self.optical_blur_sigma_sensor_pixels,
            self.full_well_electrons,
            self.read_noise_electrons,
            self.sensor_analog_gain_relative,
            self.pwm_duty_cycle,
            self.pwm_off_level,
            self.sensor_row_interval_s,
            self.pwm_phase_cycles,
            self.isp_luma_sharpen_amount,
            self.isp_luma_sharpen_sigma_pixels,
        )
        if not np.isfinite(scalars).all():
            raise ValueError("screen and sensor parameters must be finite")
        if self.exposure_electrons_per_unit is not None and not np.isfinite(
            self.exposure_electrons_per_unit
        ):
            raise ValueError("exposure must be finite")
        if self.electron_rate_per_unit_s is not None and (
            not np.isfinite(self.electron_rate_per_unit_s)
            or self.electron_rate_per_unit_s < 0
        ):
            raise ValueError("electron rate must be finite and nonnegative")
        if (
            self.exposure_electrons_per_unit is not None
            and self.electron_rate_per_unit_s is not None
        ):
            raise ValueError("choose either integrated electrons or electron rate")
        if not 0 < self.fill_fraction <= 1:
            raise ValueError("fill_fraction must be in (0, 1]")
        if self.emitter_layout not in ("vertical_rgb", "co_spatial_rgb_control"):
            raise ValueError("unsupported emitter_layout")
        if self.sensor_bayer_pattern not in ("RGGB", "BGGR", "GRBG", "GBRG"):
            raise ValueError("unsupported sensor Bayer pattern")
        if self.display_gamma <= 0 or self.optical_blur_sigma_sensor_pixels < 0:
            raise ValueError("display gamma must be positive and blur nonnegative")
        if self.diffraction_f_number is not None:
            if (
                not np.isfinite(self.diffraction_f_number)
                or self.diffraction_f_number <= 0
                or self.sensor_pixel_pitch_um is None
                or not np.isfinite(self.sensor_pixel_pitch_um)
                or self.sensor_pixel_pitch_um <= 0
            ):
                raise ValueError(
                    "diffraction requires positive f-number and sensor pixel pitch"
                )
        focus_fields = (
            self.lens_focal_length_mm,
            self.aperture_f_number,
            self.screen_distance_m,
            self.focus_distance_m,
        )
        if self.defocus_psf_model not in ("geometric_airy", "wave"):
            raise ValueError("defocus_psf_model must be geometric_airy or wave")
        if self.defocus_psf_model == "wave" and not all(
            value is not None for value in focus_fields
        ):
            raise ValueError("wave defocus PSF requires complete thin-lens parameters")
        if any(value is not None for value in focus_fields):
            if (
                any(
                    value is None or not np.isfinite(value) or value <= 0
                    for value in focus_fields
                )
                or self.sensor_pixel_pitch_um is None
                or not np.isfinite(self.sensor_pixel_pitch_um)
                or self.sensor_pixel_pitch_um <= 0
            ):
                raise ValueError(
                    "thin-lens defocus requires positive focal length, aperture, distances and pixel pitch"
                )
            focal_m = self.lens_focal_length_mm / 1000
            if self.screen_distance_m <= focal_m or self.focus_distance_m <= focal_m:
                raise ValueError(
                    "screen and focus distances must exceed lens focal length"
                )
            if self.diffraction_f_number is not None and not np.isclose(
                self.diffraction_f_number, self.aperture_f_number, rtol=0, atol=1e-12
            ):
                raise ValueError(
                    "diffraction and thin-lens aperture f-numbers must agree"
                )
            if not np.allclose(transform[2, :2], 0, rtol=0, atol=1e-12):
                raise ValueError(
                    "single-distance thin-lens defocus requires affine projection"
                )
        if self.throughput_reference_f_number is not None:
            if (
                not np.isfinite(self.throughput_reference_f_number)
                or self.throughput_reference_f_number <= 0
                or self.aperture_f_number is None
            ):
                raise ValueError(
                    "relative aperture throughput requires positive reference and current f-numbers"
                )
        if (
            len(self.rgb_effective_wavelengths_nm) != 3
            or not np.isfinite(self.rgb_effective_wavelengths_nm).all()
            or min(self.rgb_effective_wavelengths_nm) <= 0
        ):
            raise ValueError(
                "RGB effective wavelengths must be three finite positive values"
            )
        spectral_mix = np.asarray(self.sensor_spectral_mix_rgb, dtype=np.float64)
        if (
            spectral_mix.shape != (3, 3)
            or not np.isfinite(spectral_mix).all()
            or np.any(spectral_mix < 0)
            or np.any(spectral_mix.sum(axis=1) <= 0)
        ):
            raise ValueError(
                "sensor spectral mix must be a nonnegative finite 3x3 matrix with positive rows"
            )
        object.__setattr__(
            self,
            "sensor_spectral_mix_rgb",
            tuple(tuple(float(value) for value in row) for row in spectral_mix),
        )
        if (
            (
                self.exposure_electrons_per_unit is not None
                and self.exposure_electrons_per_unit < 0
            )
            or self.full_well_electrons <= 0
            or self.read_noise_electrons < 0
        ):
            raise ValueError(
                "exposure/read noise must be nonnegative and full well positive"
            )
        if self.sensor_analog_gain_relative < 1:
            raise ValueError("relative analog gain must be at least one")
        if (
            self.exposure_electrons_per_unit is None
            and self.electron_rate_per_unit_s is None
            and self.read_noise_electrons > 0
        ):
            raise ValueError(
                "read noise requires numeric exposure; None is noiseless preview"
            )
        if not 0 < self.pwm_duty_cycle <= 1 or not 0 <= self.pwm_off_level <= 1:
            raise ValueError("PWM duty must be in (0, 1] and off level in [0, 1]")
        if self.sensor_row_interval_s < 0 or not 0 <= self.pwm_phase_cycles < 1:
            raise ValueError("row interval must be nonnegative and PWM phase in [0, 1)")
        if self.isp_luma_sharpen_amount < 0 or self.isp_luma_sharpen_sigma_pixels <= 0:
            raise ValueError(
                "ISP luma sharpening amount must be nonnegative and sigma positive"
            )
        if self.exposure_time_s is not None and (
            not np.isfinite(self.exposure_time_s) or self.exposure_time_s <= 0
        ):
            raise ValueError("exposure_time_s must be positive and finite")
        if self.pwm_frequency_hz is not None and (
            not np.isfinite(self.pwm_frequency_hz)
            or self.pwm_frequency_hz <= 0
            or self.exposure_time_s is None
        ):
            raise ValueError("PWM requires positive frequency and exposure time")
        if self.electron_rate_per_unit_s is not None and self.exposure_time_s is None:
            raise ValueError("electron rate requires exposure_time_s")
        transform.setflags(write=False)
        object.__setattr__(self, "sensor_to_display", transform.copy())
        self.sensor_to_display.setflags(write=False)


@dataclass(frozen=True)
class ScreenCaptureResult:
    """Stage outputs; floating arrays are unencoded and have no JPEG artifacts."""

    emitter_band_irradiance: np.ndarray
    irradiance: np.ndarray
    noiseless_mosaic: np.ndarray
    raw_mosaic: np.ndarray
    srgb_before_sharpen: np.ndarray
    srgb: np.ndarray
    row_exposure_gain: np.ndarray
