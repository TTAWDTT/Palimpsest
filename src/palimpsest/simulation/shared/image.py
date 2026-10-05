"""Image primitives shared by digital, display and print events.

These functions do not choose event parameters or read origin labels.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .units import UM_PER_INCH

RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def rgb_unit_float(image: np.ndarray) -> np.ndarray:
    """Validate nonempty RGB; convert uint8 or unit floats to float32."""
    data = np.asarray(image)
    if data.ndim != 3 or data.shape[2] != 3 or min(data.shape[:2]) <= 0:
        raise ValueError("input must be nonempty HxWx3 RGB")
    if data.dtype == np.uint8:
        return data.astype(np.float32) / 255.0
    data = data.astype(np.float32)
    if not np.isfinite(data).all() or data.min() < 0 or data.max() > 1:
        raise ValueError("RGB float input must be finite within [0,1]")
    return data


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode unit sRGB values to linear light."""
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Clip linear light to [0,1] and encode as float32 sRGB."""
    linear = np.clip(linear, 0, 1)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1 / 2.4) - 0.055,
    ).astype(np.float32)


def blur_physical(image: np.ndarray, sigma_um: float, ppi: float) -> np.ndarray:
    """Gaussian spatial blur specified in micrometres; preserve channels."""
    if sigma_um <= 0:
        return image
    sigma = sigma_um * ppi / UM_PER_INCH
    spatial_sigma = (sigma, sigma, 0) if image.ndim == 3 else (sigma, sigma)
    return gaussian_filter(image, spatial_sigma, mode="reflect").astype(np.float32)
