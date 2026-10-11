"""Known lattices require exact tile coverage, alignment and valid local crops."""

import math
import numpy as np
import pytest

from palimpsest.detection.baselines.spatial_tiling import feature_tiles


def test_aligned_lattice_coverage():
    for height, width in ((1, 1), (97, 133), (513, 517), (2264, 3500)):
        counts = np.zeros((math.ceil(height/8), math.ceil(width/8)), int)
        for (top, bottom, left, right), (y0, y1, x0, x1) in feature_tiles(height, width):
            assert top % 8 == left % 8 == 0
            assert 0 <= y0 < y1 <= math.ceil((bottom-top)/8)
            assert 0 <= x0 < x1 <= math.ceil((right-left)/8)
            counts[top//8+y0:top//8+y1, left//8+x0:left//8+x1] += 1
            assert max(bottom-top, right-left) <= 640
        np.testing.assert_array_equal(counts, np.ones_like(counts))


def test_tile_geometry_refusals():
    for kwargs in ({'halo': 63}, {'cells': 0}, {'stride': 0}, {'height': 0}):
        args = dict(height=100, width=100)
        args.update(kwargs)
        with pytest.raises(ValueError):
            list(feature_tiles(**args))
