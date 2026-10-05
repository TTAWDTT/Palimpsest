"""Gate capture against signed real features, then time one fixed kernel head."""

import argparse
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.compiled_kernel import CompiledKernelRule
from palimpsest.detection.algorithms.kernel_readout import KernelRule
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.detection.representations.captured_clip import CapturedFrozenClip
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.detection.representations.readout import FeatureReadoutDetector
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.semantic_kernel.run_iteration import inputs
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS

OUTPUT = WORK_DIR/'robust_statistics/semantic_kernel'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    destination = OUTPUT/('capture_benchmark.json' if args.benchmark else 'capture_live_parity.json')
    if destination.exists():
        raise FileExistsError('Preserve capture evidence')
    control_path = OUTPUT/'capture_controls.json'
    control = json.loads(control_path.read_text())
    if not control['passed'] or any(file_sha256(REPO_ROOT/k) != v for k,v in control['code_pins'].items()):
        raise ValueError('Synthetic capture gate changed/failed')
    rows,_ = inputs()
    old_path = WORK_DIR/'robust_statistics/frozen_clip/benchmark.json'
    old = json.loads(old_path.read_text())
    names = {r['filename'] for r in old['measurements']}
    raw = sorted([r for r in rows if r['variant']=='raw' and r['filename'] in names], key=lambda r:r['filename'])
    if len(raw)!=120 or len(old['measurements'])!=360:
        raise ValueError('Frozen timing denominator changed')
    encoder = CapturedFrozenClip(MODELS_ROOT/'d3/ViT-L-14.pt')
    pins = {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in [Path(__file__),
            REPO_ROOT/'src/palimpsest/detection/representations/captured_clip.py',
            REPO_ROOT/'src/palimpsest/detection/algorithms/compiled_kernel.py']}
    cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        if not args.benchmark:
            selected = list(raw)
            for variant in VARIANTS[1:]:
                selected.extend(sorted([r for r in rows if r['role']=='selection' and r['variant']==variant],
                                       key=lambda r:r['filename'])[:30])
            measurements=[]
            parameters = dict(zip(VARIANTS[1:], ((90,0),(70,2),(60,2))))
            for r in selected:
                path=image_path(r)
                if file_sha256(path)!=r['sha256']:
                    raise ValueError('Capture source digest changed')
                with Image.open(path) as im:
                    pixels=np.asarray(im.convert('RGB'))
                if r['variant']!='raw':
                    stream=BytesIO();q,s=parameters[r['variant']]
                    Image.fromarray(resize256(pixels)).save(stream,format='JPEG',quality=q,subsampling=s)
                    stream.seek(0)
                    with Image.open(stream) as im:pixels=np.asarray(im.convert('RGB'))
                actual=encoder.extract(pixels).values
                expected=np.array([float(r[n]) for n in FEATURE_NAMES])
                delta=float(np.max(np.abs(actual-expected)))
                measurements.append({'filename':r['filename'],'variant':r['variant'],'maximum_difference':delta})
                if not np.array_equal(actual,expected):
                    write_json(destination,{'passed':False,'native_vectors':len(measurements),'measurements':measurements,
                                            'code_pins':pins,'scope':'Real equality refused;no tolerance widening'})
                    raise ValueError('Capture changed real feature values')
            write_json(destination,{'passed':True,'native_vectors':len(selected),'measurements':measurements,
                        'code_pins':pins,'encoder':encoder.provenance,'controls_sha256':file_sha256(control_path),
                        'timing_reference_sha256':file_sha256(old_path),'scope':'Real output equality,not invariance validation'})
            print(json.dumps({'passed':True,'native_vectors':len(selected),'maximum_difference':0}),flush=True)
            return
        parity_path=OUTPUT/'capture_live_parity.json'
        parity=json.loads(parity_path.read_text())
        if not parity['passed'] or parity['native_vectors']!=210 or parity['code_pins']!=pins:
            raise ValueError('Capture real gate changed/failed')
        iteration_path=OUTPUT/'iteration.json';iteration=json.loads(iteration_path.read_text())
        if iteration['interpretation_refused'] or not iteration['development_chosen']:
            raise ValueError('No diagnostic kernel candidate')
        for relative,sha in iteration['code_pins'].items():
            if file_sha256(REPO_ROOT/relative)!=sha:raise ValueError('Kernel fitting code changed')
        chosen=iteration['development_chosen'];rule_path=OUTPUT/(chosen.replace('/','_')+'_rule.json')
        if file_sha256(rule_path)!=iteration['rule_files'][chosen]:raise ValueError('Kernel rule changed')
        rule=CompiledKernelRule(KernelRule.load(rule_path))
        detector=FeatureReadoutDetector(encoder.extract,rule,feature_names=FEATURE_NAMES,
                                       name='captured frozen CLIP plus fixed Fourier development head')
        detector.predict(np.full((256,256,3),128,np.uint8))
        scores=rule.score([[float(r[n]) for n in FEATURE_NAMES] for r in raw])
        items=[{'filename':r['filename'],'path':image_path(r),'sha256':r['sha256'],'expected_score':float(s)}
               for r,s in zip(raw,scores)]
        result=benchmark_files(detector,items,repeats=3,seed=20261006)
        result['scope']='CUDA captured batch1 official fp16;CPU threads1;stored orientation;digest pre-read;decode+preprocess+encoder+kernel;startup excluded;no phone/localization claim'
        write_json(destination,{**result,'chosen':chosen,'encoder':encoder.provenance,'code_pins':pins,
                   'iteration_sha256':file_sha256(iteration_path),'rule_sha256':file_sha256(rule_path),
                   'live_parity_sha256':file_sha256(parity_path),'timing_reference_sha256':file_sha256(old_path)})
        print(json.dumps(result['summaries']),flush=True)


if __name__=='__main__':main()
