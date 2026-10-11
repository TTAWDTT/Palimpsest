"""Metadata-only CLI must hash its own coalescer, regardless of caller cwd."""

import json
import os
from pathlib import Path
import subprocess
import sys

from palimpsest.io.hashing import file_sha256


def test_range_plan_from_foreign_cwd_ignores_decoy_source(tmp_path):
    root = Path(__file__).resolve().parents[2]
    inventory = [
        {'name': f'source_{i // 4:04d}_{i % 4}.jpg',
         'local_offset': i * 200, 'compressed_bytes': 100}
        for i in range(3200)
    ]
    inventory_path = tmp_path/'inventory.json'
    inventory_path.write_text(json.dumps(inventory), encoding='utf-8')
    plan = {'inventory_sha256': file_sha256(inventory_path),
        'source_count': 800, 'image_count': 3200, 'archive_bytes': 640000,
        'entries': inventory, 'source_list_sha256': 'synthetic-fixture'}
    plan_path = tmp_path/'sealed_plan.json'
    plan_path.write_text(json.dumps(plan), encoding='utf-8')
    cwd = tmp_path/'foreign_directory'; cwd.mkdir()
    decoy = cwd/'src/palimpsest/io/range_plan.py'
    decoy.parent.mkdir(parents=True)
    decoy.write_text('Unrelated caller-local file\n', encoding='utf-8')
    output = tmp_path/'range_plan.json'
    result = subprocess.run([
        sys.executable, str(root/'experiments/data_preparation/deepshield/prepare_range_plan.py'),
        '--inventory', str(inventory_path), '--sealed-plan', str(plan_path), '--output', str(output),
    ], cwd=cwd, env={**os.environ, 'PYTHONPATH': str(root/'src')},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(output.read_text())
    assert receipt['coalescer_sha256'] == file_sha256(root/'src/palimpsest/io/range_plan.py')
    assert receipt['coalescer_sha256'] != file_sha256(decoy)
    assert receipt['compressed_payload_bytes'] == 320000
    assert receipt['complete_indexed_entry_bytes'] == 640000
    assert receipt['pixels_fetched'] == 0 and receipt['classifier_bound'] is False
    assert all(choice['spans'] == [[0, 639999]] for choice in receipt['choices'])
