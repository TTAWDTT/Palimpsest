"""Public screen-to-sensor renderer.

Read render_screen_capture for stage order. The implementation is separated
into _screen/parameters, emission, optics, spatial and temporal; sensor.py
contains the Bayer/ISP primitives shared with the paper-camera model.
All parameters remain hypotheses until verified against real devices.

Private helper aliases remain available for existing experiment imports.
"""

import numpy as np

from ._screen.emission import _display_emitter_raster as _display_emitter_raster
from ._screen.emission import _screen_radiance as _screen_radiance
from ._screen.emission import _screen_radiance_projective as _screen_radiance_projective
from ._screen.optics import _airy_kernel as _airy_kernel
from ._screen.optics import _airy_radius_sensor_pixels as _airy_radius_sensor_pixels
from ._screen.optics import _defocus_disk_blur as _defocus_disk_blur
from ._screen.optics import _diffraction_blur as _diffraction_blur
from ._screen.optics import (
    _effective_diffraction_f_number as _effective_diffraction_f_number,
)
from ._screen.optics import _optical_blur as _optical_blur
from ._screen.optics import _optical_halo_sensor_pixels as _optical_halo_sensor_pixels
from ._screen.optics import _wave_defocus_blur as _wave_defocus_blur
from ._screen.optics import (
    thin_lens_coc_radius_sensor_pixels as thin_lens_coc_radius_sensor_pixels,
)
from ._screen.optics import (
    thin_lens_frontoparallel_sensor_to_display as thin_lens_frontoparallel_sensor_to_display,
)
from ._screen.parameters import ScreenCaptureParameters as ScreenCaptureParameters
from ._screen.parameters import ScreenCaptureResult as ScreenCaptureResult
from ._screen.spatial import _axis_emitter_matrix as _axis_emitter_matrix
from ._screen.spatial import _emitter_pixel_weight as _emitter_pixel_weight
from ._screen.spatial import _homography_jacobian as _homography_jacobian
from ._screen.spatial import _integrate_display_raster as _integrate_display_raster
from ._screen.spatial import _is_axis_aligned_positive as _is_axis_aligned_positive
from ._screen.spatial import (
    _minimum_samples_for_projection as _minimum_samples_for_projection,
)
from ._screen.spatial import _normal_cdf_antiderivative as _normal_cdf_antiderivative
from ._screen.spatial import _spatial_axis_analytic as _spatial_axis_analytic
from ._screen.spatial import _spatial_prefilter as _spatial_prefilter
from ._screen.spatial import _spatial_tile as _spatial_tile
from ._screen.spatial import _spatial_wave_prefilter as _spatial_wave_prefilter
from ._screen.spatial import integrate_screen_radiance
from ._screen.temporal import _pwm_row_gain as _pwm_row_gain
from .sensor import _bayer_masks as _bayer_masks
from .sensor import _demosaic_bilinear as _demosaic_bilinear
from .sensor import _isp_luma_unsharp as _isp_luma_unsharp
from .sensor import _linear_to_srgb as _linear_to_srgb

__all__ = [
    "ScreenCaptureParameters",
    "ScreenCaptureResult",
    "render_screen_capture",
    "thin_lens_coc_radius_sensor_pixels",
    "thin_lens_frontoparallel_sensor_to_display",
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
    masks = _bayer_masks((height, width), parameters.sensor_bayer_pattern)
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

    linear_rgb = _demosaic_bilinear(raw_mosaic, masks)
    srgb_before_sharpen = _linear_to_srgb(linear_rgb)
    return ScreenCaptureResult(
        emitter_band_irradiance=emitter_band_irradiance,
        irradiance=irradiance,
        noiseless_mosaic=noiseless_mosaic,
        raw_mosaic=raw_mosaic,
        srgb_before_sharpen=srgb_before_sharpen,
        srgb=_isp_luma_unsharp(
            srgb_before_sharpen,
            parameters.isp_luma_sharpen_amount,
            parameters.isp_luma_sharpen_sigma_pixels,
        ),
        row_exposure_gain=row_exposure_gain,
    )
