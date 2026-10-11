"""Planted image identity and collision controls before development decoding."""

from hashlib import sha256

import numpy as np
from PIL import Image
import pytest

from experiments.data_preparation.development_content.audit_content import fingerprint,summarize,verify_counts


def row(src,value,role='fit',label='REAL'):
    return {'domain':'toy','src':src,'role':role,'label':label,**value}


def test_same_pixels_different_png_bytes_and_counts(tmp_path):
    rng=np.random.default_rng(7);pixels=rng.integers(0,256,(64,64,3),dtype=np.uint8)
    a,b=tmp_path/'a.png',tmp_path/'b.png'
    Image.fromarray(pixels).save(a,compress_level=0);Image.fromarray(pixels).save(b,compress_level=9)
    assert sha256(a.read_bytes()).digest()!=sha256(b.read_bytes()).digest()
    one,two=fingerprint(a),fingerprint(b)
    assert one==two
    result=summarize([row('a',one),row('b',two,'selection','FAKE')])
    assert result['exact_rgb_group_count']==1
    assert result['exact_rgb_cross_role_groups']==1
    assert result['exact_rgb_conflicting_label_groups']==1
    assert result['phash_candidate_count']==1
    verify_counts(result,{'exact_rgb_group_count':1})
    with pytest.raises(ValueError):verify_counts(result,{'exact_rgb_group_count':0})


def test_hash_candidate_is_not_identity_and_duplicate_key_refused(tmp_path):
    a,b=tmp_path/'dark.png',tmp_path/'light.png'
    Image.new('RGB',(64,64),(0,0,0)).save(a);Image.new('RGB',(64,64),(255,255,255)).save(b)
    one,two=fingerprint(a),fingerprint(b)
    result=summarize([row('a',one),row('b',two)])
    assert result['exact_rgb_group_count']==0
    assert result['phash_candidate_count']==1
    assert not result['phash_candidate_pairs'][0]['exact_rgb']
    with pytest.raises(ValueError):summarize([row('a',one),row('a',two)])
    with pytest.raises(ValueError):summarize([row('a',one,role='unknown')])
