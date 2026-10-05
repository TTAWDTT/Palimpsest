"""Shared Bayer sampling, bilinear demosaicing and basic ISP primitives."""

import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def bayer_masks(shape: tuple[int, int], pattern: str = "RGGB") -> np.ndarray:
    rows, columns = np.indices(shape)
    masks = np.zeros(shape + (3,), dtype=np.float32)
    layout = np.asarray(
        {
            "RGGB": ((0, 1), (1, 2)),
            "BGGR": ((2, 1), (1, 0)),
            "GRBG": ((1, 0), (2, 1)),
            "GBRG": ((1, 2), (0, 1)),
        }[pattern]
    )
    channel = layout[rows % 2, columns % 2]
    for index in range(3):
        masks[..., index] = channel == index
    return masks


def demosaic_bilinear(raw: np.ndarray, masks: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    channels = []
    for channel in range(3):
        numerator = convolve(raw * masks[..., channel], kernel, mode="reflect")
        denominator = convolve(masks[..., channel], kernel, mode="reflect")
        channels.append(numerator / np.maximum(denominator, 1e-8))
    return np.stack(channels, axis=-1)


def isp_luma_unsharp(
    srgb: np.ndarray, amount: float, sigma_pixels: float
) -> np.ndarray:
    """Optional post-tone luma edge enhancement; not a calibrated camera ISP."""
    if amount == 0:
        return srgb
    luma = srgb[..., 0] * 0.2126 + srgb[..., 1] * 0.7152 + srgb[..., 2] * 0.0722
    lowpass = gaussian_filter(luma, sigma_pixels, mode="reflect")
    delta = amount * (luma - lowpass)
    return np.clip(srgb + delta[..., None], 0, 1).astype(np.float32)
