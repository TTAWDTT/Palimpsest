"""Independent preprocessing/unit-vector controls; no torch/model downloads."""

import numpy as np
from PIL import Image
import pytest

from palimpsest.detection.representations.frozen_clip import prepare224, unit_feature, shifted_unit_feature


def test_clip_preprocessing_known_rgb_and_rectangular_crop():
    image = np.zeros((32, 16, 3), np.uint8); image[..., 0] = 255; image[..., 1] = 128
    values = prepare224(image)
    expected = (np.array([1, 128/255, 0]) - [.48145466, .4578275, .40821073]) / [.26862954, .26130258, .27577711]
    np.testing.assert_allclose(values[:, 100, 100], expected, atol=3e-7, rtol=0)
    assert values.shape == (3, 224, 224)
    image = np.arange(3*43*57, dtype=np.uint8).reshape(43, 57, 3)
    # Independent PIL construction locks truncation/center rounding on odd sizes.
    reference = np.asarray(Image.fromarray(image).resize((296, 224), Image.Resampling.BICUBIC).crop((36, 0, 260, 224)))
    actual = prepare224(image)*np.array([.26862954, .26130258, .27577711], np.float32)[:, None, None]
    actual += np.array([.48145466, .4578275, .40821073], np.float32)[:, None, None]
    np.testing.assert_array_equal(np.rint(actual*255).astype(np.uint8), reference.transpose(2, 0, 1))
    with pytest.raises(ValueError): prepare224(np.zeros((3, 3, 3), float))


def test_clip_known_unit_vector_and_refusal():
    vector = np.r_[3., 4., np.zeros(766)]
    value = unit_feature(vector)
    np.testing.assert_array_equal(value, np.r_[.6, .8, np.zeros(766)])
    assert np.linalg.norm(value) == 1
    stored = shifted_unit_feature(vector)
    np.testing.assert_allclose(2*stored-1, value, atol=1e-15, rtol=0)
    assert np.all((stored >= 0) & (stored <= 1))
    for bad in (np.zeros(768), np.full(768, np.nan), np.ones(767)):
        with pytest.raises(ValueError): unit_feature(bad)
