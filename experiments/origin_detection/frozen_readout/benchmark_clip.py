"""Decode-inclusive CUDA timing for the already selected frozen CLIP readout."""

import json
import random
import platform
import numpy as np

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES, FrozenClip
from palimpsest.detection.representations.readout import FeatureReadoutDetector
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR, REPO_ROOT, MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json

OUTPUT=WORK_DIR/'robust_statistics/frozen_clip'


def main():
    destination=OUTPUT/'benchmark.json'
    if destination.exists(): raise FileExistsError(destination)
    iteration=json.loads((OUTPUT/'iteration.json').read_text(encoding='utf-8'))
    if iteration['development_chosen']!='clip/both': raise ValueError('Benchmark only the selected clip-only candidate')
    rule_path=OUTPUT/'clip_both_rule.json'
    if file_sha256(rule_path)!=iteration['rule_files']['clip/both']: raise ValueError('Readout changed')
    receipt=json.loads((OUTPUT/'features.json').read_text(encoding='utf-8'))
    if file_sha256(OUTPUT/'features.csv')!=receipt['csv_sha256']: raise ValueError('Features changed')
    # Verify every pinned extraction artifact; the timing program is separate.
    for relative,sha in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/relative)!=sha: raise ValueError('Extraction code changed')
    rows=[r for r in read_rows(OUTPUT/'features.csv') if r['variant']=='raw']
    rule=StableRule.load(rule_path)
    previous=WORK_DIR/'robust_statistics/rr_first_iteration'
    prior=json.loads((previous/'benchmark.json').read_text(encoding='utf-8'))
    if file_sha256(previous/'benchmark.csv')!=prior['csv_sha256']: raise ValueError('Prior RR benchmark selection changed')
    names=sorted({'rr/'+r['filename'] for r in read_rows(previous/'benchmark.csv')})
    by_name={r['filename']:r for r in rows}; selected=[by_name[n] for n in names]
    if len(selected)!=60: raise ValueError('Prior RR denominator differs')
    rng=random.Random(20261006)
    for condition in ('original','mac_iphone','lg_blackfly'):
        for label in ('REAL','FAKE'):
            pool=sorted([r for r in rows if r['domain']=='chimera' and r['role']=='selection'
                         and r['condition']==condition and r['label']==label], key=lambda r:r['filename'])
            selected.extend(rng.sample(pool,10))
    if len(selected)!=120 or len({r['filename'] for r in selected})!=120: raise ValueError('Timing denominator differs')
    encoder=FrozenClip(MODELS_ROOT/'d3/ViT-L-14.pt')
    detector=FeatureReadoutDetector(encoder.extract,rule,feature_names=FEATURE_NAMES,name='frozen CLIP plus group/pair conventional readout')
    detector.predict(np.full((256,256,3),128,np.uint8))
    scores=rule.score([[float(r[n]) for n in FEATURE_NAMES] for r in selected])
    items=[{'filename':r['filename'],'path':image_path(r),'sha256':r['sha256'],'expected_score':float(s)} for r,s in zip(selected,scores)]
    result=benchmark_files(detector,items,repeats=3,seed=20261006)
    result['scope']='CUDA batch1,stored orientation,file digest pre-read/warm filesystem cache,decode+preprocess+encoder+readout;startup/warmup excluded;no phone/localization claim'
    write_json(destination,{**result,'encoder':encoder.provenance,'iteration_sha256':file_sha256(OUTPUT/'iteration.json'),
                           'rule_sha256':file_sha256(rule_path),'prior_rr_timing_selection_sha256':prior['csv_sha256'],
                           'code_sha256':file_sha256(REPO_ROOT/'experiments/origin_detection/frozen_readout/benchmark_clip.py'),
                           'adapter_sha256':file_sha256(REPO_ROOT/'src/palimpsest/detection/representations/readout.py'),
                           'processor':platform.processor(),'python':platform.python_version()})
    print(json.dumps(result['summaries']),flush=True)


if __name__=='__main__':main()
