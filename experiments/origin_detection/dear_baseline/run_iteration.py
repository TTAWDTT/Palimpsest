"""One strict official DEAR-r development comparison; no fitting or tuning."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
from threadpoolctl import threadpool_limits

from palimpsest.detection.baselines.dear import DearRDetector, TiledDearRDetector
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.robust_views import evaluate_views
from palimpsest.evaluation.scored_pixels import score_inventory
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS

OUTPUT = WORK_DIR / 'robust_statistics/dear_baseline'
VENDOR = WORK_DIR / 'robust_statistics/asymmetry_review/code'


def source_pins():
    names = ['dear/__init__.py', 'dear/detector/__init__.py', 'dear/nn_classifier/__init__.py',
             'dear/detector/rajan_mask_gated_detector.py',
             'dear/detector/corvi_mask_gated_detector.py', 'dear/nn_classifier/resnet.py']
    tree = json.loads((VENDOR.parent / 'code_tree.json').read_text())
    blobs = {r['path']: r['sha'] for r in tree['tree'] if r['type'] == 'blob'}
    for name in names:
        raw = (VENDOR / name).read_bytes()
        git_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if git_blob != blobs[name]:
            raise ValueError('DEAR source does not match official pinned Git blob')
    return {name: file_sha256(VENDOR / name) for name in names}


def code_pins():
    paths = [Path(__file__), REPO_ROOT / 'experiments/origin_detection/dear_baseline/README.md',
             REPO_ROOT / 'src/palimpsest/detection/baselines/dear.py',
             REPO_ROOT / 'src/palimpsest/evaluation/scored_pixels.py',
             REPO_ROOT / 'src/palimpsest/evaluation/robust_views.py',
             REPO_ROOT / 'src/palimpsest/detection/baselines/spatial_tiling.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--tiled', action='store_true')
    args = parser.parse_args()
    base = OUTPUT.with_name('dear_baseline_tiled') if args.tiled else OUTPUT
    output = base.with_name(base.name + '_pilot') if args.pilot else base
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'iteration.json').exists() or (output / 'scores.csv').exists():
        raise FileExistsError('Preserve DEAR evidence')
    if args.tiled:
        gate = json.loads((OUTPUT / 'real_tiling_controls.json').read_text())
        synthetic = json.loads((OUTPUT / 'tiling_controls.json').read_text())
        if (not gate['passed'] or not synthetic['passed']
                or any(file_sha256(REPO_ROOT/k) != v for k,v in synthetic['code_pins'].items())
                or gate['script_sha256'] != file_sha256(REPO_ROOT / 'experiments/origin_detection/dear_baseline/audit_real_tiling.py')
                or gate['synthetic_receipt_sha256'] != file_sha256(OUTPUT / 'tiling_controls.json')):
            raise ValueError('Tiled execution gate changed')
    _, inventory, parent, _ = parent_data()
    inventory = sorted((r for r in inventory if r['role'] == 'selection'), key=lambda r: r['filename'])
    if len(inventory) != 1260:
        raise ValueError('DEAR selection denominator differs')
    if args.pilot:
        groups = defaultdict(list)
        for record in inventory:
            groups[record['domain'], record['label'], record['condition']].append(record)
        selected = {r['filename']: r for g in groups.values() for r in g[:4]}
        largest = sorted(inventory, key=lambda r: int(r['width']) * int(r['height']), reverse=True)
        for r in largest[:6]:
            selected[r['filename']] = r
        for r in largest:
            if len(selected) == 60:
                break
            selected[r['filename']] = r
        inventory = sorted(selected.values(), key=lambda r: r['filename'])
    else:
        pilot_path = base.with_name(base.name + '_pilot') / 'iteration.json'
        pilot = json.loads(pilot_path.read_text())
        if not pilot['passed'] or pilot['code_pins'] != code_pins() or pilot['source_pins'] != source_pins():
            raise ValueError('DEAR pilot/source gate changed')
    cv2.setNumThreads(1)
    detector_type = TiledDearRDetector if args.tiled else DearRDetector
    detector = detector_type(VENDOR, MODELS_ROOT / 'dear/dear_r.pth', source_pins=source_pins())
    with threadpool_limits(limits=1):
        rows, receipt = score_inventory(inventory, detector, resolve_path=image_path, resize=resize256,
              variants=VARIANTS, jpeg_parameters=dict(zip(VARIANTS[1:], ((90,0),(70,2),(60,2)))),
              progress=lambda v: print(json.dumps(v), flush=True))
        result = None if args.pilot else evaluate_views(rows, variant_order=VARIANTS)
    with (output / 'scores.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'iteration.json', {**receipt, 'passed': True, 'result': result,
        'source_pins': source_pins(), 'code_pins': code_pins(), 'encoder': detector.provenance,
        'inventory_sha256': parent['inventory_sha256'], 'scores_sha256': file_sha256(output / 'scores.csv'),
        'scope': 'Official fixed neural baseline, exposed selection, not independent validation',
        'execution_variant': 'aligned tiled' if args.tiled else 'original native',
        'parameters_fitted': 0, 'goal_achieved': False})
    if result:
        print(json.dumps({k: result[k] for k in ('minimum_domain_ba', 'minimum_scene_ba',
                           'maximum_any_scene_drop', 'worst_pair')}), flush=True)


if __name__ == '__main__':
    main()
