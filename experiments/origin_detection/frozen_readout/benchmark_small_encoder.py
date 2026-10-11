"""Same120 native files as stage16, for one already chosen small rule."""

import csv
from dataclasses import replace
import json
import platform
from pathlib import Path
import numpy as np

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.representations.frozen_dinov2_small import FEATURE_NAMES,FrozenDinoV2Small
from palimpsest.detection.representations.readout import FeatureReadoutDetector
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR,REPO_ROOT,MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import image_path,write_json

OUTPUT=WORK_DIR/'robust_statistics/compact_frozen_encoder'


def main():
    destination=OUTPUT/'benchmark.json'
    if destination.exists():raise FileExistsError('Preserve small native timing')
    iteration_path=OUTPUT/'iteration.json';iteration=json.loads(iteration_path.read_text())
    chosen=iteration['development_chosen']
    if chosen is None or iteration['interpretation_refused']:raise ValueError('No accepted diagnostic candidate')
    rule_path=OUTPUT/(chosen.replace('/','_')+'_rule.json')
    if file_sha256(rule_path)!=iteration['rule_files'][chosen]:raise ValueError('Chosen rule changed')
    receipt=json.loads((OUTPUT/'features.json').read_text())
    if receipt['csv_sha256']!=file_sha256(OUTPUT/'features.csv'):raise ValueError('Small cache changed')
    for relative,sha in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/relative)!=sha:raise ValueError('Small extraction code changed')
    old_path=WORK_DIR/'robust_statistics/frozen_clip/benchmark.json'
    old=json.loads(old_path.read_text());names={r['filename'] for r in old['measurements']}
    if len(names)!=120 or len(old['measurements'])!=360:raise ValueError('Frozen comparison timing files differ')
    with (OUTPUT/'features.csv').open(encoding='utf-8',newline='') as stream:
        selected=[r for r in csv.DictReader(stream) if r['variant']=='raw' and r['filename'] in names]
    if len(selected)!=120 or {r['filename'] for r in selected}!=names:raise ValueError('Timing/cache coverage differs')
    rule=StableRule.load(rule_path)
    if rule.feature_names!=FEATURE_NAMES[:len(rule.feature_names)]:raise ValueError('Unexpected small-rule descriptor subset')
    encoder=FrozenDinoV2Small(MODELS_ROOT/'dinov2_small/dinov2_vits14_pretrain.pth',
                            WORK_DIR/'robust_statistics/compact_encoder_review/runtime_receipt.json')
    def extract(image):
        features=encoder.extract(image)
        return replace(features,values=features.values[:len(rule.feature_names)])
    detector=FeatureReadoutDetector(extract,rule,feature_names=rule.feature_names,name='frozen DINOv2-S plus convex development head')
    detector.predict(np.full((256,256,3),128,np.uint8))
    scores=rule.score([[float(r[n]) for n in rule.feature_names] for r in selected])
    items=[{'filename':r['filename'],'path':image_path(r),'sha256':r['sha256'],'expected_score':float(s)} for r,s in zip(selected,scores)]
    result=benchmark_files(detector,items,repeats=3,seed=20261006)
    result['scope']='CUDA batch1 float32 TF32disabled;stored orientation;digest pre-read/warm file cache;decode+preprocess+encoder+readout;startup excluded;no phone/localization claim'
    write_json(destination,{**result,'chosen':chosen,'encoder':encoder.provenance,
                'iteration_sha256':file_sha256(iteration_path),'rule_sha256':file_sha256(rule_path),
                'timing_reference_sha256':file_sha256(old_path),'source_sha256':file_sha256(Path(__file__)),
                'benchmark_module_sha256':file_sha256(REPO_ROOT/'src/palimpsest/evaluation/file_benchmark.py'),
                'adapter_sha256':file_sha256(REPO_ROOT/'src/palimpsest/detection/representations/readout.py'),
                'python':platform.python_version(),'processor':platform.processor()})
    print(json.dumps(result['summaries']),flush=True)


if __name__=='__main__':main()
