"""Known constant image through real JPEG/array writing, plus corrupt witnesses."""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from palimpsest.evaluation.numeric_view_increment import extract_numeric_view
from palimpsest.io.hashing import file_sha256


def test_complete_increment_and_corrupt_reference(tmp_path):
    image = tmp_path/'flat.png'; Image.fromarray(np.full((12, 16, 3), 128, np.uint8)).save(image)
    inventory = [{'filename': 'flat', 'role': 'fit', 'sha256': file_sha256(image), 'width': 16, 'height': 12}]
    def extract(path):
        with Image.open(path) as im: pixels = np.asarray(im.convert('RGB'), float)
        return SimpleNamespace(values=np.array([pixels[0, 0, 0], pixels.std()]), probability_fake=.25,
                               elapsed_ms=1., statistics_ms=0., diagnostics={})
    def validate(rows, matrix, native):
        assert len(rows) == 1 and native == inventory
        assert np.array_equal(matrix, [[128., 0.]]) and rows[0]['variant'] == 'q60'
    kwargs = dict(variant='q60', quality=60, sampling=2, resolve_path=lambda row: image,
                  resize=lambda pixels: pixels, validate=validate)
    encoder = SimpleNamespace(extract_file=extract)
    out = tmp_path/'good'
    result = extract_numeric_view(inventory, out, ('constant', 'std'), encoder,
        reference={('flat', 'q60'): (np.array([128., 0.]), .25)}, **kwargs)
    assert result['complete_reference_exact'] and result['records'] == 1
    assert np.array_equal(np.load(out/'vectors.npy'), [[128., 0.]]) and not (out/'query.jpg').exists()
    with pytest.raises(ValueError, match='changed signed parent'):
        extract_numeric_view(inventory, tmp_path/'bad', ('constant', 'std'), encoder,
            reference={('flat', 'q60'): (np.array([128.001, 0.]), .25)}, **kwargs)
    assert (tmp_path/'bad/vectors.partial.npy').exists() and not (tmp_path/'bad/vectors.npy').exists()
    with pytest.raises(ValueError, match='duplicate'):
        extract_numeric_view(inventory*2, tmp_path/'duplicate', ('constant', 'std'), encoder, **kwargs)
