"""Public screen-to-sensor renderer.

Read render_screen_capture for stage order. The implementation is separated
into _render/parameters, emission, optics, spatial and temporal; shared/sensor.py
contains the Bayer/ISP primitives shared with the paper-camera model.
All parameters remain hypotheses until verified against real devices.

Helpers are imported directly from their owning implementation modules.
"""

import numpy as np

from palimpsest.simulation.screen_capture._render.parameters import (
    ScreenCaptureParameters as ScreenCaptureParameters,
)
from palimpsest.simulation.screen_capture._render.parameters import (
    ScreenCaptureResult as ScreenCaptureResult,
)
from palimpsest.simulation.screen_capture._render.spatial import (
    integrate_screen_radiance as integrate_screen_radiance,
)
from palimpsest.simulation.screen_capture._render.temporal import _pwm_row_gain
from palimpsest.simulation.shared.sensor import bayer_masks
from palimpsest.simulation.shared.sensor import demosaic_bilinear
from palimpsest.simulation.shared.sensor import isp_luma_unsharp
from palimpsest.simulation.shared.image import linear_to_srgb

__all__ = [
    "ScreenCaptureParameters",
    "ScreenCaptureResult",
    "render_screen_capture",
]


def render_screen_capture(
    frame: np.ndarray,
    sensor_shape: tuple[int, int],
    parameters: ScreenCaptureParameters,
    *,
    seed: int = 0,
    samples_per_sensor_pixel: int | None = None,
    tile_size_sensor_pixels: int | None = None,
    spatial_method: str = "fine",
) -> ScreenCaptureResult:
    """Render display, projective optics, sensor integration, RAW and simple ISP.

    The output is an unencoded crop. Optional temporal PWM/rolling exposure
    uses one square-wave global screen luminance signal and row start times.
    The optional fixed spectral mix is an effective display-primary to sensor
    channel response, not measured spectral sensitivity. Measured PSF, lens
    distortion, real ISP and JPEG remain outside this model. Thin-lens defocus
    is a uniform-distance circular-aperture approximation and couples the
    aperture to Airy diffraction. Relative pupil-area throughput is opt-in
    through throughput_reference_f_number at fixed focus/distance. Set
    defocus_psf_model="wave" for an ideal scalar through-focus circular-pupil
    reference instead of sequential Airy and geometric disk kernels. Optional
    luma sharpening is an ISP hypothesis. Parameters are virtual until fitted.
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[2] != 3 or not np.isfinite(frame).all():
        raise ValueError("frame must be a finite HxWx3 array")
    if frame.size == 0 or frame.min() < 0 or frame.max() > 1:
        raise ValueError("frame values must lie in [0, 1]")
    if len(sensor_shape) != 2 or any(
        int(size) != size or size <= 0 for size in sensor_shape
    ):
        raise ValueError("sensor_shape must contain two positive integer dimensions")
    height, width = map(int, sensor_shape)
    # Display response, optics and area integration precede sensor readout.
    emitted_frame = np.power(frame, parameters.display_gamma)
    irradiance = integrate_screen_radiance(
        emitted_frame,
        (height, width),
        parameters,
        spatial_method=spatial_method,
        samples_per_sensor_pixel=samples_per_sensor_pixel,
        tile_size_sensor_pixels=tile_size_sensor_pixels,
    )
    # Integrate time, then apply the optional pupil-area throughput factor.
    row_exposure_gain = _pwm_row_gain(height, parameters)
    irradiance *= row_exposure_gain[:, None, None]
    if parameters.throughput_reference_f_number is not None:
        # Calibrated electron gain is defined at the reference f-number.
        # At fixed focus/distance, ideal pupil area gives a (Nref/N)^2 ratio.
        # Transmission losses, field vignetting and exposure compensation are
        # intentionally separate, uncalibrated effects.
        irradiance *= (
            parameters.throughput_reference_f_number / parameters.aperture_f_number
        ) ** 2

    emitter_band_irradiance = irradiance
    spectral_mix = np.asarray(parameters.sensor_spectral_mix_rgb, dtype=np.float32)
    if not np.array_equal(spectral_mix, np.eye(3, dtype=np.float32)):
        irradiance = np.einsum(
            "hwd,cd->hwc", emitter_band_irradiance, spectral_mix, optimize=True
        ).astype(np.float32)

    # CFA sampling -> photon/read noise -> analog gain -> basic ISP.
    masks = bayer_masks((height, width), parameters.sensor_bayer_pattern)
    mosaiced_irradiance = (irradiance * masks).sum(axis=-1)
    noiseless_mosaic = mosaiced_irradiance.copy()
    raw_mosaic = noiseless_mosaic.copy()
    electrons_per_unit = parameters.exposure_electrons_per_unit
    if parameters.electron_rate_per_unit_s is not None:
        electrons_per_unit = (
            parameters.electron_rate_per_unit_s * parameters.exposure_time_s
        )
    if electrons_per_unit is not None:
        expected_electrons = np.maximum(mosaiced_irradiance, 0) * electrons_per_unit
        # Shot noise is drawn from collected photoelectrons. Analog gain acts
        # only on the readout signal, with a unity-gain-equivalent ADC ceiling;
        # it does not create photons or improve photon shot-noise SNR.
        gain = parameters.sensor_analog_gain_relative
        noiseless_mosaic = np.clip(
            expected_electrons * gain / parameters.full_well_electrons, 0, 1
        ).astype(np.float32)
        rng = np.random.default_rng(seed)
        electrons = rng.poisson(expected_electrons)
        if parameters.read_noise_electrons > 0:
            electrons = electrons + rng.normal(
                0, parameters.read_noise_electrons, electrons.shape
            )
        raw_mosaic = np.clip(
            electrons * gain / parameters.full_well_electrons, 0, 1
        ).astype(np.float32)

    linear_rgb = demosaic_bilinear(raw_mosaic, masks)
    srgb_before_sharpen = linear_to_srgb(linear_rgb)
    return ScreenCaptureResult(
        emitter_band_irradiance=emitter_band_irradiance,
        irradiance=irradiance,
        noiseless_mosaic=noiseless_mosaic,
        raw_mosaic=raw_mosaic,
        srgb_before_sharpen=srgb_before_sharpen,
        srgb=isp_luma_unsharp(
            srgb_before_sharpen,
            parameters.isp_luma_sharpen_amount,
            parameters.isp_luma_sharpen_sigma_pixels,
        ),
        row_exposure_gain=row_exposure_gain,
    )
