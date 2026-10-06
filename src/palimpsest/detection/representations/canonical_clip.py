"""Fixed resize/JPEG normalization before the unchanged CLIP representation.

Repeated JPEG is lossy and not an idempotent invariant projection. Any benefit
must be measured on actual matched processed images.
"""

from io import BytesIO
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.detection.algorithms.residual_statistics.features import resize256
from .frozen_clip import ClipFeatures, FrozenClip


def canonical_pixels(image):
    image = resize256(image)
    stream = BytesIO()
    Image.fromarray(image).save(stream, format='JPEG', quality=70, subsampling=2)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return np.asarray(decoded.convert('RGB')).copy()


class CanonicalClip:
    def __init__(self, checkpoint, *, device='cuda:0'):
        self.base = FrozenClip(checkpoint, device=device)
        self.provenance = {**self.base.provenance,
            'canonicalization': 'AREA longedge<=256 then PIL JPEG Q70 subsampling2',
            'lossy_not_idempotent': True, 'parameters_selected_from_query': False}

    def extract(self, image):
        start = perf_counter()
        pixels = canonical_pixels(image)
        added = 1000 * (perf_counter() - start)
        result = self.base.extract(pixels)
        return ClipFeatures(result.values, added + result.preprocess_ms, result.statistics_ms)
