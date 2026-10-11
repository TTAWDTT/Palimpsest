"""Reject changed signed prefix/probability/query while allowing new coordinates."""

import numpy as np
import pytest

from experiments.origin_detection.cure_token_quantiles_full.prepare_features import validate_q60_row


def test_signed_q60_extension_corruption_controls():
    old = np.arange(3600, dtype=float)
    new = np.r_[old, np.arange(320, dtype=float)]
    row = {'probability_fake': .3, 'query_sha256': 'known_query'}
    previous = {'probability_fake': '.3', 'query_sha256': 'known_query'}
    validate_q60_row(row, new, previous, old)
    changed = new.copy(); changed[-1] += 1
    validate_q60_row(row, changed, previous, old)
    changed[0] += 1
    with pytest.raises(ValueError):
        validate_q60_row(row, changed, previous, old)
    for bad in ({**row, 'probability_fake': .31}, {**row, 'query_sha256': 'other_query'}):
        with pytest.raises(ValueError):
            validate_q60_row(bad, new, previous, old)
