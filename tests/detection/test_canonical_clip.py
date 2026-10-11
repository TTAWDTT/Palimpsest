"""Known flat identity and a counterexample to JPEG idempotence."""

import numpy as np
import pytest

from palimpsest.detection.representations.canonical_clip import canonical_pixels


def test_known_constant_geometry_and_channel_order():
    flat = np.full((512, 256, 3), 128, dtype=np.uint8)
    result = canonical_pixels(flat)
    assert result.shape == (256, 128, 3)
    assert np.array_equal(result, np.full(result.shape, 128, dtype=np.uint8))
    red = np.full((32, 32, 3), (240, 20, 20), dtype=np.uint8)
    result = canonical_pixels(red)
    assert result.shape == red.shape
    assert result[:, :, 0].mean() > result[:, :, 1].mean() + 200


def test_repeat_deterministic_but_not_idempotent():
    image = np.random.default_rng(7).integers(0, 256, (83, 97, 3), dtype=np.uint8)
    once = canonical_pixels(image)
    assert np.array_equal(once, canonical_pixels(image))
    assert not np.array_equal(once, canonical_pixels(once))
    with pytest.raises(ValueError):
        canonical_pixels(image.astype(float))
