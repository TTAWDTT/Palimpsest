"""Tests for stage ordering and codec effects in the publication operator."""

import numpy as np
import pytest

from palimpsest.simulation.digital.publication import PublicationParameters, apply_publication


def test_crop_before_resize_and_lossless_png() -> None:
    source = np.zeros((8, 12, 3), dtype=np.uint8)
    source[2:6, 3:9] = (200, 100, 50)
    result = apply_publication(
        source, PublicationParameters(crop_xyxy=(3, 2, 9, 6), output_size=(3, 2))
    )
    assert result.after_crop_rgb.shape == (4, 6, 3)
    assert result.after_resize_rgb.shape == (2, 3, 3)
    assert result.encoded_bytes.startswith(b"\x89PNG")
    np.testing.assert_array_equal(result.decoded_rgb, result.after_resize_rgb)
    np.testing.assert_array_equal(
        result.decoded_rgb, np.full((2, 3, 3), (200, 100, 50))
    )


def test_jpeg_requires_explicit_chroma_and_changes_decoded_pixels() -> None:
    source = np.zeros((32, 32, 3), dtype=np.uint8)
    source[:, ::2] = (255, 0, 0)
    source[:, 1::2] = (0, 0, 255)
    params = PublicationParameters(encoding="jpeg", jpeg_quality=70, jpeg_subsampling=2)
    result = apply_publication(source, params)
    assert result.encoded_bytes.startswith(b"\xff\xd8")
    assert not np.array_equal(result.decoded_rgb, result.after_resize_rgb)
    assert apply_publication(source, params).encoded_bytes == result.encoded_bytes
    with pytest.raises(ValueError):
        PublicationParameters(encoding="jpeg", jpeg_quality=70)


def test_invalid_capture_and_crop_rejected() -> None:
    with pytest.raises(ValueError):
        apply_publication(np.zeros((4, 4)), PublicationParameters())
    with pytest.raises(ValueError):
        apply_publication(
            np.zeros((4, 4, 3), dtype=np.uint8),
            PublicationParameters(crop_xyxy=(0, 0, 5, 4)),
        )
    with pytest.raises(ValueError):
        PublicationParameters(encoding="png", jpeg_quality=95)
