"""Frozen B-Free representation, gated synthetic and native parity/cost pilot."""

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.detection.representations.frozen_bfree import FEATURE_NAMES, FrozenBFree
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS

OUTPUT = WORK_DIR/'robust_statistics/bfree_readout'
VENDOR = WORK_DIR/'vendor/bfree/code'
REVISION = 'c6a9f898782fb466b29af01f21960b67415afb0e'


def source_pins():
    actual = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=VENDOR.parent,
                            capture_output=True, text=True, check=True).stdout.strip()
    if actual != REVISION:
        raise ValueError('Pinned B-Free vendor revision changed')
    pins = {}
    for name in ('networks/__init__.py', 'networks/wrapper5crops.py', 'utils/normalization.py', '../LICENSE.txt'):
        git_path = 'LICENSE.txt' if name.startswith('../') else 'code/'+name
        expected = subprocess.run(['git', 'rev-parse', REVISION+':'+git_path], cwd=VENDOR.parent,
                                  capture_output=True, text=True, check=True).stdout.strip()
        data = (VENDOR/name).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual != expected:
            raise ValueError('B-Free file differs from signed Git blob')
        pins[name] = file_sha256(VENDOR/name)
    return pins


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (
        'src/palimpsest/detection/representations/frozen_bfree.py',
        'src/palimpsest/detection/baselines/bfree.py',
        'src/palimpsest/detection/algorithms/residual_statistics/features.py',
        'src/palimpsest/evaluation/source_readout_campaign.py',
        'src/palimpsest/evaluation/features.py', 'src/palimpsest/evaluation/source_training.py',
        'src/palimpsest/evaluation/balanced_null.py', 'src/palimpsest/evaluation/robust_views.py',
        'src/palimpsest/detection/algorithms/source_view_risk.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_frozen_bfree.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def runtime():
    import timm
    import torch
    import torchvision
    files = [Path(timm.__file__).parent/'models/vision_transformer.py',
             Path(timm.__file__).parent/'layers/patch_embed.py',
             Path(torchvision.__file__).parent/'transforms/functional.py']
    return {'versions': {k: importlib.metadata.version(k) for k in ('torch', 'torchvision', 'timm', 'numpy', 'Pillow')},
            'modules': {k: str(v) for k, v in [('torch', torch.__file__), ('timm', timm.__file__)]},
            'library_pins': {str(p): file_sha256(p) for p in files}, 'torch_threads': torch.get_num_threads(),
            'device': torch.cuda.get_device_name(0), 'dtype': 'FP32,TF32off,batch1,tile64'}


def exact_feature(first, second):
    if not np.array_equal(first.values, second.values) or first.official_logit != second.official_logit:
        raise ValueError('B-Free repeated feature/official score differs')


def synthetic_gate(encoder, output):
    arrays = {'flat': np.full((301, 417, 3), 128, np.uint8),
              'channels': np.broadcast_to([32, 128, 224], (520, 620, 3)).astype(np.uint8),
              'spatial': np.random.default_rng(20261006).integers(0, 256, (301, 417, 3), dtype=np.uint8)}
    results = {}
    for name, pixels in arrays.items():
        path = output/(name+'.png')
        if path.exists():
            raise FileExistsError('Preserve B-Free synthetic gate')
        Image.fromarray(pixels).save(path)
        first, repeat = encoder.extract_file(path), encoder.extract_file(path)
        exact_feature(first, repeat)
        direct = encoder.direct_file_logit(path)
        if direct != first.official_logit:
            raise ValueError('Hooked B-Free changed original forward score')
        try:
            exact_feature(first, replace(first, official_logit=first.official_logit+.001))
        except ValueError:
            pass
        else:
            raise ValueError('Corrupted score passed exact parity')
        results[name] = {'repeat_exact': True, 'direct_score_exact': True, 'changed_score_rejected': True,
                         'linear_mean_error': first.linear_mean_error, 'sha256': file_sha256(path)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pilot', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    output = OUTPUT.with_name(OUTPUT.name+'_pilot') if args.pilot else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    if (output/'features.csv').exists() or (output/'features.json').exists():
        raise FileExistsError('Preserve B-Free representation run')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('B-Free mean controls changed')
    old, inventory, parent, _ = parent_data()
    del old
    pilot = None
    if args.pilot:
        prior = WORK_DIR/'robust_statistics/cure_readout_pilot'
        receipt = json.loads((prior/'features.json').read_text())
        if receipt['csv_sha256'] != file_sha256(prior/'features.csv'):
            raise ValueError('Signed old pilot changed')
        names = {r['filename'] for r in read_rows(prior/'features.csv')}
        inventory = [r for r in inventory if r['filename'] in names]
        if len(inventory) != 60 or any(r['role'] != 'selection' for r in inventory):
            raise ValueError('Expected previously exposed60-file pilot')
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    packages = runtime()
    if not args.pilot:
        p = OUTPUT.with_name(OUTPUT.name+'_pilot')
        pilot = json.loads((p/'features.json').read_text())
        if (pilot['code_pins'] != code_pins() or pilot['runtime'] != packages or not pilot['cost_passed']
                or not pilot['all_repeated_exact'] or pilot['records'] != 60
                or pilot['csv_sha256'] != file_sha256(p/'features.csv')):
            raise ValueError('B-Free signed cost/parity pilot differs or failed')
    encoder = FrozenBFree(VENDOR, MODELS_ROOT/'bfree', source_pins=source_pins())
    witnesses = synthetic_gate(encoder, output) if args.pilot else pilot['synthetic_witnesses']
    rows, durations = [], []
    start = perf_counter()
    scratch = output/'query.jpg'
    if scratch.exists():
        raise FileExistsError('Unfinished query requires audit')
    try:
        for i, row in enumerate(inventory, 1):
            path = image_path(row)
            if file_sha256(path) != row['sha256']:
                raise ValueError('Native B-Free feature digest differs')
            with Image.open(path) as opened:
                if opened.size != (int(row['width']), int(row['height'])):
                    raise ValueError('Native B-Free dimensions differ')
                pixels = None if args.pilot else np.asarray(opened.convert('RGB'))
            variants = VARIANTS[:1] if args.pilot else VARIANTS if row['role'] == 'selection' else VARIANTS[:2]
            for variant in variants:
                query = path
                if variant != 'raw':
                    quality, subsampling = (90, 0) if variant == VARIANTS[1] else (70, 2) if variant == VARIANTS[2] else (60, 2)
                    Image.fromarray(resize256(pixels)).save(scratch, format='JPEG', quality=quality, subsampling=subsampling)
                    query = scratch
                feature = encoder.extract_file(query)
                durations.append(feature.end_to_end_ms)
                if args.pilot:
                    exact_feature(feature, encoder.extract_file(query))
                rows.append({**row, 'variant': variant, **dict(zip(FEATURE_NAMES, feature.values.tolist())),
                             'official_logit': feature.official_logit})
            if i % 60 == 0 or i == len(inventory):
                print(json.dumps({'images': i, 'total': len(inventory), 'records': len(rows),
                                  'elapsed_s': round(perf_counter()-start, 3)}), flush=True)
    finally:
        encoder.close()
    if scratch.exists():
        scratch.unlink()
    for role in ('fit', 'threshold', 'selection'):
        selected = [r for r in inventory if r['role'] == role]
        if selected:
            validate_feature_cache([r for r in rows if r['role'] == role], selected, FEATURE_NAMES,
                variants=VARIANTS[:1] if args.pilot else VARIANTS if role == 'selection' else VARIANTS[:2],
                bounds=(-np.inf, np.inf))
    median = float(np.median(durations))
    with (output/'features.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output/'features.json', {'images': len(inventory), 'records': len(rows),
        'elapsed_s': perf_counter()-start, 'csv_sha256': file_sha256(output/'features.csv'),
        'code_pins': code_pins(), 'runtime': packages, 'encoder': encoder.provenance,
        'inventory_sha256': parent['inventory_sha256'], 'synthetic_witnesses': witnesses,
        'all_repeated_exact': True if args.pilot else None, 'cost_median_ms': median,
        'cost_passed': median <= 500 if args.pilot else None,
        'scope': 'Frozen released B-Free vector;feature protocol,not full old baseline rerun'})
    if args.pilot and median > 500:
        raise ValueError('B-Free full extraction cost budget refused')


if __name__ == '__main__':
    main()
