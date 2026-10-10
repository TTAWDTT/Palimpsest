"""Capture synthetic gate only; no real image or fitting on failure."""

import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from palimpsest.detection.representations.captured_clip import CapturedFrozenClip
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR,MODELS_ROOT,REPO_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import write_json


def main():
    path=WORK_DIR/'robust_statistics/semantic_kernel/capture_controls.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve capture controls')
    pins={str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in [Path(__file__),
          REPO_ROOT/'src/palimpsest/detection/representations/captured_clip.py',
          REPO_ROOT/'src/palimpsest/detection/representations/frozen_clip.py',
          REPO_ROOT/'experiments/origin_detection/semantic_kernel/README.md']}
    try:
        encoder=CapturedFrozenClip(MODELS_ROOT/'d3/ViT-L-14.pt')
        elapsed=[];torch=encoder.torch
        with torch.inference_mode():
            for index in range(120):
                torch.cuda.synchronize();start=perf_counter();encoder._graph.replay();torch.cuda.synchronize()
                if index>=20:elapsed.append(1000*(perf_counter()-start))
        result={'passed':True,'encoder':encoder.provenance,'synthetic_only':True,'native_images_used':0,
                'graph_replay_only_p50_ms':float(np.quantile(elapsed,.5)),
                'graph_replay_only_p95_ms':float(np.quantile(elapsed,.95)),
                'scope':'Synthetic replay only,no CPU preprocessing/copies/head/decode;not end-to-end speed'}
    except Exception as error:
        result={'passed':False,'native_images_used':0,'error':str(error),'traceback':traceback.format_exc(),
                'scope':'Capture refused;leave original frozen encoder intact'}
    write_json(path,{**result,'code_pins':pins})
    print(json.dumps({k:v for k,v in result.items() if k not in ('encoder','traceback')}),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
