"""Twelve signed feasible native files verify tiled scores and decisions."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from palimpsest.detection.baselines.dear import TiledDearRDetector
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import MODELS_ROOT, REPO_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.source_view_risk.run_iteration import parent_data
from experiments.origin_detection.dear_baseline.run_iteration import OUTPUT, VENDOR, source_pins


def main():
    destination = OUTPUT / 'real_tiling_controls.json'
    if destination.exists():
        raise FileExistsError('Preserve real numerical gate')
    synthetic_path = OUTPUT / 'tiling_controls.json'
    synthetic = json.loads(synthetic_path.read_text())
    if not synthetic['passed'] or any(file_sha256(REPO_ROOT/k) != v for k,v in synthetic['code_pins'].items()):
        raise ValueError('Synthetic tiling gate changed')
    _, inventory, parent, _ = parent_data()
    groups = defaultdict(list)
    for record in inventory:
        if record['role'] == 'selection' and int(record['width']) * int(record['height']) <= 1_200_000:
            groups[record['domain'], record['label'], record['condition']].append(record)
    if len(groups) != 12:
        raise ValueError('Missing feasible domain/class/condition group')
    chosen = [max(g, key=lambda r: (int(r['width']) * int(r['height']), r['filename']))
              for _,g in sorted(groups.items())]
    detector = TiledDearRDetector(VENDOR, MODELS_ROOT / 'dear/dear_r.pth', source_pins=source_pins())
    measurements = []
    with threadpool_limits(limits=1), detector.torch.inference_mode():
        for record in chosen:
            path = image_path(record)
            if file_sha256(path) != record['sha256']:
                raise ValueError('Real parity file digest changed')
            with Image.open(path) as im:
                pixels = np.asarray(im.convert('RGB'))
            native = float(detector.detector.predict(detector.transform(Image.fromarray(pixels))
                           .unsqueeze(0).to(detector.device)).item())
            tiled = detector.predict(pixels).score
            difference = abs(native-tiled)
            measurements.append({'filename': record['filename'], 'sha256': record['sha256'],
                'native_score': native, 'tiled_score': tiled, 'difference': difference,
                'same_decision': (native > 0) == (tiled > 0)})
            if difference > 1e-4 or (native > 0) != (tiled > 0):
                write_json(destination, {'passed': False, 'measurements': measurements,
                    'scope': 'Real execution parity refused;no tolerance widening'})
                raise ValueError('Native/tiled real score gate refused')
    write_json(destination, {'passed': True, 'native_images': len(chosen), 'measurements': measurements,
        'maximum_difference': max(r['difference'] for r in measurements),
        'synthetic_receipt_sha256': file_sha256(synthetic_path),
        'script_sha256': file_sha256(Path(__file__)), 'inventory_sha256': parent['inventory_sha256'],
        'encoder': detector.provenance, 'scope': '12feasible real numerical checks,not alllarge-image parity or accuracy'})
    print(json.dumps({'passed': True, 'native_images': 12,
                     'maximum_difference': max(r['difference'] for r in measurements)}), flush=True)


if __name__ == '__main__':
    main()
