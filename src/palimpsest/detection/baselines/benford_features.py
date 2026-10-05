"""Bonettini-inspired adaptation: fixed Benford curve, in-memory DCT Q95.

This is not a numeric reproduction of the paper's generalized curve or JPEG
entropy-coefficient extraction. Feature code is independent of sklearn.
"""

import numpy as np
from PIL import Image
from scipy.fft import dctn
from palimpsest.contracts import RGBImage, validate_rgb

BASE_QUANTIZATION = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)
QUANTIZATION_95 = np.maximum(1, np.floor((BASE_QUANTIZATION * 10 + 50) / 100))
ZIGZAG_FIRST_NINE = (
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
)
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def crop_grayscale(loaded: Image.Image) -> np.ndarray:
    width, height = loaded.size
    short_side = min(width, height)
    left = (width - short_side) // 2
    top = (height - short_side) // 2
    square = loaded.crop((left, top, left + short_side, top + short_side))
    # A single normalized field of view across source and processed images.
    gray = square.convert("L").resize((256, 256), Image.Resampling.BICUBIC)
    return np.asarray(gray, dtype=np.float32)


def first_digit_features(gray: np.ndarray) -> np.ndarray:
    blocks = gray.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3) - 128
    coefficients = dctn(blocks, axes=(-2, -1), norm="ortho")
    quantized = np.rint(coefficients / QUANTIZATION_95)
    features = []
    for row, column in ZIGZAG_FIRST_NINE:
        values = np.abs(quantized[..., row, column].ravel())
        nonzero = values[values > 0]
        if len(nonzero):
            magnitude = np.floor(np.log10(nonzero))
            digits = np.floor(nonzero / np.power(10.0, magnitude)).astype(np.int64)
            digits = np.clip(digits, 1, 9)
            distribution = np.bincount(digits, minlength=10)[1:10] / len(nonzero)
        else:
            distribution = np.zeros(9)
        midpoint = (distribution + BENFORD) / 2
        js = 0.5 * np.sum(
            np.where(
                distribution > 0,
                distribution * np.log2(np.maximum(distribution, 1e-12) / midpoint),
                0,
            )
        )
        js += 0.5 * np.sum(BENFORD * np.log2(BENFORD / midpoint))
        features.extend((float(js), len(nonzero) / len(values)))
    result = np.asarray(features, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite DCT feature")
    return result


def image_features(image: RGBImage) -> np.ndarray:
    validate_rgb(image)
    return first_digit_features(crop_grayscale(Image.fromarray(image)))
