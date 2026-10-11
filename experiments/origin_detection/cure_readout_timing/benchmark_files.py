"""Live cost of the fixed full CuRe conventional source-risk rule."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
from time import perf_counter

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.representations.frozen_cure import FULL_NAMES, FrozenCure
from palimpsest.evaluation.timing import percentile
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.cure_baseline.run_iteration import VENDOR, source_pins, runtime
from experiments.origin_detection.cure_readout.prepare_features import OUTPUT as PARENT, code_pins as parent_pins
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json

OUTPUT = WORK_DIR / 'robust_statistics/cure_readout_timing'


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md',
        REPO_ROOT / 'src/palimpsest/detection/algorithms/paired_stability.py',
        REPO_ROOT / 'src/palimpsest/detection/representations/frozen_cure.py',
        REPO_ROOT / 'src/palimpsest/evaluation/timing.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / 'benchmark.json'
    if destination.exists():
        raise FileExistsError('Preserve fixed readout live timing')
    feature = json.loads((PARENT / 'features.json').read_text())
    iteration = json.loads((PARENT / 'iteration.json').read_text())
    rule_path = PARENT / 'full_source_rule.json'
    if (feature['code_pins'] != parent_pins() or feature['runtime'] != runtime()
            or feature['csv_sha256'] != file_sha256(PARENT / 'features.csv')
            or iteration['rule_files']['full/source'] != file_sha256(rule_path)):
        raise ValueError('Frozen CuRe representation or rule changed')
    rule = StableRule.load(rule_path)
    if rule.feature_names != FULL_NAMES:
        raise ValueError('Fixed full representation changed')
    reference = WORK_DIR / 'robust_statistics/cure_baseline/benchmark.json'
    prior = json.loads(reference.read_text())
    names = {r['filename'] for r in prior['measurements']}
    with (PARENT / 'features.csv').open(newline='', encoding='utf-8-sig') as stream:
        rows = sorted((r for r in csv.DictReader(stream) if r['variant'] == 'raw' and r['filename'] in names),
                      key=lambda r: r['filename'])
    roles = dict(Counter(r['role'] for r in rows))
    if len(rows) != 120 or len(prior['measurements']) != 360 or roles != {'fit': 32, 'selection': 74, 'threshold': 14}:
        raise ValueError('Native timing denominator changed')
    for row in rows:
        if file_sha256(image_path(row)) != row['sha256']:
            raise ValueError('Native timing pixel digest changed')
    cv2.setNumThreads(1)
    model = FrozenCure(VENDOR, VENDOR / 'weights/cure_adapter.pt', MODELS_ROOT / 'cure/PE-Core-L14-336.pt',
                       source_pins=source_pins())
    expected = {}
    with threadpool_limits(limits=1):
        for row in rows:
            cached = np.array([float(row[n]) for n in FULL_NAMES])
            live = model.extract_file(image_path(row)).values[:1024]
            if not np.array_equal(live, cached):
                raise ValueError('Live full CuRe raw vector differs')
            expected[row['filename']] = float(rule.score(cached[None])[0] - rule.threshold)
        rng = random.Random(20261006)
        measurements = []
        for repeat in range(3):
            queue = list(rows)
            rng.shuffle(queue)
            for row in queue:
                start = perf_counter()
                features = model.extract_file(image_path(row))
                ready = perf_counter()
                margin = float(rule.score(features.values[None, :1024])[0] - rule.threshold)
                decision = margin > 0
                end = perf_counter()
                if margin != expected[row['filename']] or decision != (expected[row['filename']] > 0):
                    raise ValueError('Live fixed margin or decision differs')
                measurements.append({'filename': row['filename'], 'repeat': repeat,
                    'pixels': int(row['width']) * int(row['height']),
                    'decode_preprocess_ms': features.decode_preprocess_ms, 'encoder_ms': features.encoder_ms,
                    'readout_ms': 1000 * (end-ready), 'end_to_end_ms': 1000 * (end-start)})
    groups = defaultdict(list)
    for row in measurements:
        groups['all'].append(row)
        groups['at_most_2MP' if row['pixels'] <= 2_000_000 else 'above_2MP'].append(row)
    summaries = {key: {'images': len({r['filename'] for r in selected}), 'measurements': len(selected),
        **{metric: {f'p{int(q*100)}': percentile([r[metric] for r in selected], q) for q in (.5, .95)}
           for metric in ('decode_preprocess_ms', 'encoder_ms', 'readout_ms', 'end_to_end_ms')}}
        for key, selected in groups.items()}
    write_json(destination, {'summaries': summaries, 'measurements': measurements, 'code_pins': code_pins(),
        'encoder': model.provenance, 'runtime': runtime(), 'rule_sha256': file_sha256(rule_path),
        'feature_receipt_sha256': file_sha256(PARENT / 'features.json'),
        'iteration_sha256': file_sha256(PARENT / 'iteration.json'),
        'reference_benchmark_sha256': file_sha256(reference), 'timing_only_role_counts': roles,
        'exact_raw_vectors': 120, 'exact_live_margins': 360,
        'scope': 'Warm native batch1;decode+author preprocessing+frozen encoder+traditional head;'
                 'hashes/cache audit/startup excluded;BLAS1 torch4 OpenCV1;not mobile runtime'})
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
