import numpy as np
import pytest
from palimpsest.contracts import Box, Origin, Prediction, Region, validate_rgb


def test_origin_is_original_content_and_threshold_is_strict():
    assert Prediction("fixed", 0.1, threshold=0.1).origin == Origin.NATURAL
    assert Prediction("fixed", 0.10001, threshold=0.1).origin == Origin.AI
    assert Prediction("fixed", 0.2).ai_probability is None
    with pytest.raises(ValueError, match="finite"):
        Prediction("bad", float("nan"))


def test_pixel_contract_rejects_implicit_color_or_range_conversion():
    with pytest.raises(ValueError):
        validate_rgb(np.zeros((10, 10, 3), np.float32))
    validate_rgb(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError):
        Box(0, 0, 0, 1)
    with pytest.raises(ValueError):
        Box(0, 0, 11, 10).validate_shape((10, 10))


def test_mask_coordinates_are_full_source_and_inside_box():
    mask = np.zeros((10, 12), bool)
    mask[3:6, 4:7] = True
    Region("0", Box(4, 3, 7, 6), "candidate", mask=mask).validate_shape(mask.shape)
    with pytest.raises(ValueError, match="contained"):
        Region("0", Box(5, 3, 7, 6), "candidate", mask=mask).validate_shape(mask.shape)
