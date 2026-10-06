"""Known pixels verify raw orientation, score margin and explicit encoding."""

import numpy as np
from PIL import Image
import pytest

from palimpsest.contracts import Prediction
from palimpsest.evaluation.scored_pixels import score_inventory
from palimpsest.io.hashing import file_sha256


def test_signed_pixels_and_margin(tmp_path):
    pixels = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    path = tmp_path / 'known.png'
    Image.fromarray(pixels).save(path)
    record = dict(filename=path.name, sha256=file_sha256(path), width=7, height=5)
    seen = []

    class KnownDetector:
        def predict(self, image):
            seen.append(image.copy())
            return Prediction('known', 3, 2, timing_ms={'predict_ms': 1})

    rows, receipt = score_inventory([record], KnownDetector(), resolve_path=lambda _: path,
        resize=lambda x: x[:3, :4], variants=('raw', 'jpeg'), jpeg_parameters={'jpeg': (70, 2)})
    np.testing.assert_array_equal(seen[0], pixels)
    assert seen[1].shape == (3, 4, 3)
    assert [r['score'] for r in rows] == [1, 1]
    assert [r['variant'] for r in rows] == ['raw', 'jpeg']
    assert receipt['images'] == 1 and receipt['records'] == 2
    with pytest.raises(ValueError, match='digest'):
        score_inventory([{**record, 'sha256': 'bad'}], KnownDetector(), resolve_path=lambda _: path,
            resize=lambda x: x, variants=('raw',), jpeg_parameters={})


def test_invalid_encoding_and_duplicate_inventory():
    arguments = dict(detector=None, resolve_path=None, resize=None)
    with pytest.raises(ValueError, match='Encoding'):
        score_inventory([{'filename': 'a'}], variants=('raw', 'jpeg'), jpeg_parameters={}, **arguments)
    with pytest.raises(ValueError, match='Duplicate'):
        score_inventory([{'filename': 'a'}, {'filename': 'a'}], variants=('raw',), jpeg_parameters={}, **arguments)
