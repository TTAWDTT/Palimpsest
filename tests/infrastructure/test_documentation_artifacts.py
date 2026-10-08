"""Ignored receipts must fail link portability even when present locally."""

import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('layout_artifact_check',ROOT/'tools/check_layout.py')
layout = importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)


def test_existing_private_receipt_is_not_a_portable_doc_link(tmp_path):
    if shutil.which('git') is None:pytest.skip('Git unavailable for ignore-policy test')
    subprocess.run(['git','init','-q'],cwd=tmp_path,check=True)
    (tmp_path/'.gitignore').write_text('/work/*\n!/work/README.md\n')
    work = tmp_path/'work';work.mkdir()
    private, public = work/'receipt.json',work/'README.md'
    private.write_text('{}');public.write_text('Public local-work guide')
    assert private.exists() and public.exists()
    assert layout.ignored_link_targets(tmp_path,[private,public])=={private.resolve()}
