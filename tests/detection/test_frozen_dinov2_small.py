"""Known RGB normalization/crop and descriptor schema controls; no weights."""

import numpy as np
import pytest
from palimpsest.detection.representations.frozen_dinov2_small import prepare224,shifted_units


def test_small_encoder_rgb_and_center_crop():
    image=np.full((256,512,3),[255,0,128],np.uint8)
    value=prepare224(image)
    assert value.shape==(3,224,224)
    np.testing.assert_allclose(value[:,0,0],[(1-.485)/.229,(0-.456)/.224,(128/255-.406)/.225],atol=3e-7,rtol=0)
    image=np.zeros((256,512,3),np.uint8);image[:,0:100]=255
    assert np.max(prepare224(image)[0])<0
    with pytest.raises(ValueError):prepare224(image.astype(float))


def test_small_encoder_independent_units():
    cls=np.zeros(384);cls[0]=3
    patch=np.zeros(384);patch[-1]=-2
    stored=shifted_units(cls,patch)
    assert len(stored)==768 and stored[0]==1 and stored[-1]==0
    assert np.count_nonzero(stored==.5)==766
    np.testing.assert_array_equal(shifted_units(cls*2,patch*3),stored)
    for wrong in (np.zeros(384),np.full(384,np.nan),np.ones(383)):
        with pytest.raises(ValueError):shifted_units(wrong,patch)
