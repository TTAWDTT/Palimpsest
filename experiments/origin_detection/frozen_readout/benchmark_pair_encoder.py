"""Live two-encoder scores and decode-inclusive timing on the signed120 queue."""

import csv
import json
from pathlib import Path

import cv2
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.paired_stability import StableRule
from palimpsest.detection.representations.frozen_clip import FrozenClip, FEATURE_NAMES as CLIP_NAMES
from palimpsest.detection.representations.frozen_dinov2_small import FrozenDinoV2Small, FEATURE_NAMES as DINO_NAMES
from palimpsest.detection.representations.feature_pair import FrozenFeaturePair
from palimpsest.detection.representations.readout import FeatureReadoutDetector
from palimpsest.evaluation.pixel_features import join_features
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import write_json, image_path

OUTPUT = WORK_DIR/'robust_statistics/complementary_encoders'


def selected_rows(path, names):
    with path.open(encoding='utf-8',newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['filename'] in names and r['variant']=='raw']
    if len(rows) != 120 or {r['filename'] for r in rows} != names:
        raise ValueError('Frozen pair timing coverage changed')
    return sorted(rows,key=lambda r:r['filename'])


def main():
    destination = OUTPUT/'benchmark.json'
    if destination.exists():
        raise FileExistsError('Preserve paired encoder live benchmark')
    iteration_path = OUTPUT/'iteration.json'
    iteration = json.loads(iteration_path.read_text())
    if iteration['development_chosen'] != 'joint/source' or iteration['interpretation_refused']:
        raise ValueError('No specified joint development candidate')
    for relative,sha in iteration['code_pins'].items():
        if file_sha256(REPO_ROOT/relative) != sha:
            raise ValueError('Pair computation changed')
    rule_path = OUTPUT/'joint_source_rule.json'
    if file_sha256(rule_path) != iteration['rule_files']['joint/source']:
        raise ValueError('Pair rule changed')
    reference_path = WORK_DIR/'robust_statistics/frozen_clip/benchmark.json'
    reference = json.loads(reference_path.read_text())
    names = {r['filename'] for r in reference['measurements']}
    if len(names) != 120 or len(reference['measurements']) != 360:
        raise ValueError('Timing reference changed')
    components = []
    for directory in ('frozen_clip','compact_frozen_encoder'):
        folder = WORK_DIR/'robust_statistics'/directory
        receipt = json.loads((folder/'features.json').read_text())
        if receipt['csv_sha256'] != file_sha256(folder/'features.csv'):
            raise ValueError('Pair component cache changed')
        for relative,sha in receipt['code_pins'].items():
            if file_sha256(REPO_ROOT/relative) != sha:
                raise ValueError('Pair extraction code changed')
        components.append(selected_rows(folder/'features.csv',names))
    rows = join_features(*components,DINO_NAMES)
    rule = StableRule.load(rule_path)
    cv2.setNumThreads(1)
    first = FrozenClip(MODELS_ROOT/'d3/ViT-L-14.pt')
    second = FrozenDinoV2Small(MODELS_ROOT/'dinov2_small/dinov2_vits14_pretrain.pth',
        WORK_DIR/'robust_statistics/compact_encoder_review/runtime_receipt.json')
    pair = FrozenFeaturePair(first,second,first_names=CLIP_NAMES,second_names=DINO_NAMES)
    detector = FeatureReadoutDetector(pair.extract,rule,feature_names=pair.feature_names,
        name='Frozen CLIP plus DINO source-risk development readout')
    with threadpool_limits(limits=1):
        scores = rule.score([[float(r[n]) for n in pair.feature_names] for r in rows])
        items = [dict(filename=r['filename'],path=image_path(r),sha256=r['sha256'],expected_score=float(s))
                 for r,s in zip(rows,scores)]
        result = benchmark_files(detector,items,repeats=3,seed=20261006,score_tolerance=1e-12)
    write_json(destination,{**result,'encoder':pair.provenance,
        'scope':'Live sequential2 frozen encoders,batch1;digestpreread warms files;decode+preprocess+bothencoders+head;startup excluded;no phone claim',
        'live_cache_score_tolerance':1e-12,'iteration_sha256':file_sha256(iteration_path),
        'rule_sha256':file_sha256(rule_path),'timing_reference_sha256':file_sha256(reference_path),
        'script_sha256':file_sha256(Path(__file__))})
    print(json.dumps(result['summaries']),flush=True)


if __name__ == '__main__':
    main()
