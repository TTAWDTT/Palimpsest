"""Explicit image transforms used by matched statistical diagnostics."""

import numpy as np
from PIL import Image


def resize_unit_float(image: Image.Image, side: int) -> np.ndarray:
    """Bicubic square resize, float32 divided by 255; no color conversion.

    This preserves the historical RR diagnostic transform. It is not physical
    recapture and must not be substituted for a detector's own preprocessing.
    """
    return (
        np.asarray(
            image.resize((side, side), Image.Resampling.BICUBIC), dtype=np.float32
        )
        / 255
    )
