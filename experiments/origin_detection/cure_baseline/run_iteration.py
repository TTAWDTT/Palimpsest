"""Strict official CuRe fixed baseline on the existing exposed selection queue."""

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.detection.baselines.cure import CureFileDetector
from palimpsest.evaluation.robust_views import evaluate_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT, PATHS
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS

OUTPUT = WORK_DIR / 'robust_statistics/cure_baseline'
VENDOR = PATHS.workspace / 'external/cure'
TREE = WORK_DIR / 'robust_statistics/recent_response_review/cure_source/tree.json'
REVISION = '8f7832c83dcd3f4f89de955103e94a64ddc3cdc9'


def source_pins():
    tree = json.loads(TREE.read_text(encoding='utf-8'))
    if tree['commit'] != REVISION or tree['tree'].get('truncated'):
        raise ValueError('Unexpected CuRe source tree')
    blobs = {r['path']: r['sha'] for r in tree['tree']['tree'] if r['type'] == 'blob'}
    names = ['cure/__init__.py', 'cure/backbone.py', 'cure/model.py', 'cure/preprocess.py',
             'infer.py', 'requirements.txt', 'LICENSE', 'NOTICE']
    pins = {}
    for name in names:
        raw = (VENDOR / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if actual != blobs[name]:
            raise ValueError('CuRe file differs from pinned official Git blob')
        pins[name] = file_sha256(VENDOR / name)
    return pins


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md',
        REPO_ROOT / 'src/palimpsest/detection/baselines/cure.py',
        REPO_ROOT / 'tests/detection/test_cure_margin.py',
        REPO_ROOT / 'src/palimpsest/evaluation/robust_views.py',
        REPO_ROOT / 'src/palimpsest/evaluation/pairing.py',
        REPO_ROOT / 'src/palimpsest/evaluation/classification.py',
        REPO_ROOT / 'src/palimpsest/detection/algorithms/residual_statistics/features.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def runtime():
    versions = {k: importlib.metadata.version(k) for k in
                ('torch', 'numpy', 'Pillow', 'scipy', 'opencv-python-headless', 'albumentations', 'albucore')}
    if versions['albumentations'] != '2.0.8':
        raise ValueError('CuRe needs pinned author albumentations2.0.8')
    return versions


def synthetic_gate(detector, directory):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261006)
    rgb = rng.integers(0, 256, (417, 301, 3), dtype=np.uint8)
    png, jpeg96 = directory / 'random.png', directory / 'random96.jpg'
    if png.exists() or jpeg96.exists():
        raise FileExistsError('Preserve CuRe synthetic controls')
    Image.fromarray(rgb).save(png)
    Image.fromarray(rgb).save(jpeg96, quality=96)
    first, second = detector.preprocess(png), detector.preprocess(jpeg96)
    if not detector.torch.equal(first, second) or not detector.torch.equal(first, detector.preprocess(png)):
        raise ValueError('Author PNG96 branch or preprocessing repeat differs')
    flat = directory / 'flat.jpg'
    Image.fromarray(np.full((301, 417, 3), 128, np.uint8)).save(flat, quality=100)
    flat_tensor = detector.preprocess(flat)
    expected = (128 / 255 - .5) / .5
    if first.shape != (3, 336, 336) or not np.allclose(flat_tensor.numpy(), expected, rtol=0, atol=2e-7):
        raise ValueError('CuRe synthetic dimensions/flat normalization failed')
    prediction = detector.predict_file(png)
    repeated = detector.predict_file(png)
    with detector.torch.inference_mode(), detector.torch.amp.autocast('cuda', enabled=True):
        direct = float(detector.model(first.unsqueeze(0).to(detector.device)).softmax(dim=1)[0, 1].float().cpu())
    if prediction.probability_fake != repeated.probability_fake or prediction.probability_fake != direct:
        raise ValueError('CuRe wrapper/direct/repeat probabilities differ')
    return {'passed': True, 'png_branch_exact': True, 'preprocess_repeat_exact': True,
            'flat_normalization': True, 'shape': list(first.shape), 'model_repeat_exact': True,
            'direct_author_probability_exact': True, 'synthetic_probability': direct,
            'synthetic_files': {p.name: file_sha256(p) for p in (png, jpeg96, flat)}}


def detector():
    return CureFileDetector(VENDOR, VENDOR / 'weights/cure_adapter.pt',
                            MODELS_ROOT / 'cure/PE-Core-L14-336.pt', source_pins=source_pins())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    output = OUTPUT.with_name(OUTPUT.name + '_pilot') if args.pilot else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'iteration.json').exists() or (output / 'scores.csv').exists():
        raise FileExistsError('Preserve CuRe scores')
    packages = runtime()
    software_path = OUTPUT / 'software_controls.json'
    software = json.loads(software_path.read_text())
    if not software['passed'] or any(file_sha256(REPO_ROOT / k) != v for k, v in software['test_pins'].items()):
        raise ValueError('CuRe software gate differs')
    if not args.pilot:
        pilot = json.loads((OUTPUT.with_name(OUTPUT.name + '_pilot') / 'iteration.json').read_text())
        if (not pilot['passed'] or pilot['code_pins'] != code_pins() or pilot['source_pins'] != source_pins()
                or pilot['runtime'] != packages):
            raise ValueError('CuRe pilot/source/runtime gate differs')
    _, inventory, parent, _ = parent_data()
    inventory = sorted((r for r in inventory if r['role'] == 'selection'), key=lambda r: r['filename'])
    if len(inventory) != 1260:
        raise ValueError('CuRe selection coverage differs')
    if args.pilot:
        groups = defaultdict(list)
        for record in inventory:
            groups[(record['domain'], record['scene'], record['label'], record['condition'])].append(record)
        selected = {r['filename']: r for g in groups.values() for r in g[:1]}
        for r in sorted(inventory, key=lambda r: int(r['width']) * int(r['height']), reverse=True):
            if len(selected) >= 60:
                break
            selected[r['filename']] = r
        inventory = sorted(selected.values(), key=lambda r: r['filename'])
    cv2.setNumThreads(1)
    model = detector()
    gate = synthetic_gate(model, output / 'synthetic')
    scratch = output / 'query.jpg'
    if scratch.exists():
        raise FileExistsError('Unfinished CuRe query exists; audit before continuing')
    parameters = dict(zip(VARIANTS[1:], ((90, 0), (70, 2), (60, 2))))
    rows, start = [], perf_counter()
    for index, record in enumerate(inventory, 1):
        native = image_path(record)
        if file_sha256(native) != record['sha256']:
            raise ValueError('CuRe pixel source changed')
        with Image.open(native) as image:
            if image.size != (int(record['width']), int(record['height'])):
                raise ValueError('CuRe pixel dimensions changed')
            pixels = np.asarray(image.convert('RGB'))
        for variant in VARIANTS:
            path = native
            if variant != 'raw':
                q, s = parameters[variant]
                Image.fromarray(resize256(pixels)).save(scratch, format='JPEG', quality=q, subsampling=s)
                path = scratch
            prediction = model.predict_file(path)
            rows.append({**record, 'variant': variant, 'score': prediction.margin,
                'probability_fake': prediction.probability_fake,
                'decode_preprocess_ms': prediction.decode_preprocess_ms,
                'forward_ms': prediction.forward_ms, 'end_to_end_ms': prediction.end_to_end_ms})
        if index % 30 == 0 or index == len(inventory):
            print(json.dumps({'images': index, 'total': len(inventory), 'records': len(rows),
                              'elapsed_s': round(perf_counter()-start, 3)}), flush=True)
    scratch.unlink()
    result = None if args.pilot else evaluate_views(rows, variant_order=VARIANTS)
    with (output / 'scores.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'iteration.json', {'images': len(inventory), 'records': len(rows),
        'elapsed_s': perf_counter()-start, 'passed': True, 'result': result,
        'code_pins': code_pins(), 'source_pins': source_pins(), 'encoder': model.provenance,
        'runtime': packages, 'synthetic_gate': gate, 'software_controls_sha256': file_sha256(software_path),
        'inventory_sha256': parent['inventory_sha256'], 'scores_sha256': file_sha256(output / 'scores.csv'),
        'parameters_fitted': 0, 'goal_achieved': False,
        'scope': 'Official CuRe batch1 file protocol;exposed selection;not independent validation'})
    if result:
        print(json.dumps({k: result[k] for k in ('minimum_domain_ba', 'minimum_scene_ba',
                                               'maximum_any_scene_drop', 'worst_pair')}), flush=True)


if __name__ == '__main__':
    main()
