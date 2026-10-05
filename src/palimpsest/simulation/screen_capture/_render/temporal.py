"""Integrate the assumed display PWM over each sensor row exposure."""

import numpy as np

from palimpsest.simulation.screen_capture._render.parameters import (
    ScreenCaptureParameters,
)


def _pwm_row_gain(height: int, parameters: ScreenCaptureParameters) -> np.ndarray:
    """Fraction of peak-on display radiance collected during each row exposure.

    The display input and electron-rate parameters refer to the emitter's
    *on-state* radiance. Lower duty therefore collects fewer photons over a
    complete PWM cycle; no compensation for perceived brightness is assumed.
    """
    if parameters.pwm_frequency_hz is None:
        return np.ones(height, dtype=np.float32)
    frequency = parameters.pwm_frequency_hz
    duty = parameters.pwm_duty_cycle
    duration = parameters.exposure_time_s
    phase = parameters.pwm_phase_cycles
    row_start_cycles = (
        phase
        + np.arange(height, dtype=np.float64)
        * parameters.sensor_row_interval_s
        * frequency
    )
    row_end_cycles = row_start_cycles + duration * frequency

    def bright_cycles(t: np.ndarray) -> np.ndarray:
        whole = np.floor(t)
        return whole * duty + np.minimum(t - whole, duty)

    bright_fraction = (
        bright_cycles(row_end_cycles) - bright_cycles(row_start_cycles)
    ) / (duration * frequency)
    gain = parameters.pwm_off_level + (1 - parameters.pwm_off_level) * bright_fraction
    return gain.astype(np.float32)
