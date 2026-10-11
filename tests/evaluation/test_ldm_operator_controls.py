"""Pinned operator assay refuses changed source before optional Torch imports."""

import pytest

from experiments.origin_detection.non_neural.generator_mechanisms.audit_ldm_operator import _pinned_class, assay


def test_changed_source_refused_before_execution(tmp_path):
    source=tmp_path/'changed.py'
    source.write_text('raise RuntimeError("must never execute")\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Source SHA'):
        _pinned_class(source)


def test_unregistered_dimensions_refused_before_source_or_torch(tmp_path):
    with pytest.raises(ValueError,match='fixture length'):
        assay(tmp_path/'missing.py',100)
