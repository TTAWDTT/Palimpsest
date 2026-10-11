"""Synthetic preprocessing, strict model/repeat, and rough model-only cost."""

import json
import importlib.util
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.detection.representations.frozen_dinov2_small import FrozenDinoV2Small,prepare224
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR,MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import write_json


def main():
    path=WORK_DIR/'robust_statistics/compact_frozen_encoder/runtime_controls.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError('Preserve runtime controls')
    receipt=WORK_DIR/'robust_statistics/compact_encoder_review/runtime_receipt.json'
    encoder=FrozenDinoV2Small(MODELS_ROOT/'dinov2_small/dinov2_vits14_pretrain.pth',receipt)
    # Import official eval transforms only after the pinned package is installed.
    spec=importlib.util.spec_from_file_location('official_dinov2_transforms',encoder.vendor/'dinov2/data/transforms.py')
    transforms=importlib.util.module_from_spec(spec);spec.loader.exec_module(transforms)
    rng=np.random.default_rng(20261006)
    differences=[]
    for h,w in [(256,256),(173,351),(352,174)]:
        image=rng.integers(0,256,(h,w,3),dtype=np.uint8)
        expected=transforms.make_classification_eval_transform()(Image.fromarray(image)).numpy()
        actual=prepare224(image)
        if not np.array_equal(actual,expected): raise ValueError('Official preprocessing differs')
        differences.append(float(np.max(np.abs(actual-expected))))
    tensor=encoder.torch.linspace(-1,1,3*224*224,device=encoder.device).reshape(1,3,224,224)
    elapsed=[]
    with encoder.torch.inference_mode():
        for index in range(120):
            encoder.torch.cuda.synchronize();start=perf_counter()
            encoder.model.forward_features(tensor);encoder.torch.cuda.synchronize()
            if index>=20:elapsed.append(1000*(perf_counter()-start))
    write_json(path,{'passed':True,'synthetic_only':True,'native_images_used':0,
                    'preprocessing_exact':True,'maximum_preprocessing_differences':differences,
                    'encoder':encoder.provenance,'forward_only_p50_ms':float(np.quantile(elapsed,.5)),
                    'forward_only_p95_ms':float(np.quantile(elapsed,.95)),
                    'scope':'Synthetic fixed tensor warm CUDA forward only;not image-to-decision or end-to-end speed',
                    'source_receipt_sha256':file_sha256(receipt),
                    'script_sha256':file_sha256(WORK_DIR.parent/'experiments/origin_detection/compact_frozen_encoder/audit_runtime.py')})
    print(json.dumps({'passed':True,'forward_p95_ms':float(np.quantile(elapsed,.95))}),flush=True)


if __name__=='__main__':main()
