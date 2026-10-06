"""Time one signed smaller CLIP head against the original 120-file queue."""

import json
from pathlib import Path

import cv2
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.paired_stability import StableRule
from palimpsest.detection.representations.frozen_clip_base import FEATURE_NAMES, FrozenClipBase
from palimpsest.detection.representations.readout import FeatureReadoutDetector
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import MODELS_ROOT, REPO_ROOT, WORK_DIR
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from palimpsest.io.tables import read_rows

OUTPUT = WORK_DIR / 'robust_statistics/compact_semantic_encoder'


def main():
    destination = OUTPUT / 'benchmark.json'
    if destination.exists():
        raise FileExistsError('Preserve base timing receipt')
    iteration_path = OUTPUT / 'iteration.json'
    iteration = json.loads(iteration_path.read_text())
    receipt_path = OUTPUT / 'features.json'
    receipt = json.loads(receipt_path.read_text())
    if (iteration['interpretation_refused'] or not iteration['development_chosen']
            or iteration['features_receipt_sha256'] != file_sha256(receipt_path)
            or receipt['csv_sha256'] != file_sha256(OUTPUT / 'features.csv')):
        raise ValueError('Base evidence changed or was refused')
    for relative, sha in iteration['code_pins'].items():
        if file_sha256(REPO_ROOT / relative) != sha:
            raise ValueError('Base computation code changed')
    chosen = iteration['development_chosen']
    rule_path = OUTPUT / (chosen + '_rule.json')
    if file_sha256(rule_path) != iteration['rule_files'][chosen]:
        raise ValueError('Base rule changed')
    rule = StableRule.load(rule_path)
    reference_path = WORK_DIR / 'robust_statistics/frozen_clip/benchmark.json'
    reference = json.loads(reference_path.read_text())
    names = {r['filename'] for r in reference['measurements']}
    raw = sorted((r for r in read_rows(OUTPUT / 'features.csv')
                  if r['variant'] == 'raw' and r['filename'] in names),
                 key=lambda r: r['filename'])
    if len(raw) != 120 or len(reference['measurements']) != 360:
        raise ValueError('Timing queue changed')
    cv2.setNumThreads(1)
    encoder = FrozenClipBase(MODELS_ROOT / 'clip_small/ViT-B-16.pt')
    detector = FeatureReadoutDetector(encoder.extract, rule, feature_names=FEATURE_NAMES,
                                    name='Frozen CLIP-B/16 conventional source-risk head')
    with threadpool_limits(limits=1):
        scores = rule.score([[float(r[n]) for n in FEATURE_NAMES] for r in raw])
        items = [dict(filename=r['filename'], path=image_path(r), sha256=r['sha256'],
                      expected_score=float(s)) for r, s in zip(raw, scores)]
        result = benchmark_files(detector, items, repeats=3, seed=20261006)
    result['scope'] = ('Batch1 official frozen fp16;CPU threads1;stored orientation;'
                       'digest pre-read;decode+preprocess+encoder+head;startup excluded;'
                       'no mobile/localization claim;failed accuracy candidate cost only')
    write_json(destination, {**result, 'chosen': chosen, 'encoder': encoder.provenance,
               'iteration_sha256': file_sha256(iteration_path),
               'rule_sha256': file_sha256(rule_path),
               'timing_reference_sha256': file_sha256(reference_path),
               'script_sha256': file_sha256(Path(__file__))})
    print(json.dumps(result['summaries']), flush=True)


if __name__ == '__main__':
    main()
