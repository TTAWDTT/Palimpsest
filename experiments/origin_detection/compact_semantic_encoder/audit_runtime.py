"""Original pinned transform helpers and archive repeat controls, no real pixels."""

import ast
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.detection.representations.frozen_clip import prepare224
from palimpsest.detection.representations.frozen_clip_base import FrozenClipBase
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR,MODELS_ROOT,REPO_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT=WORK_DIR/'robust_statistics/compact_semantic_encoder'
SOURCE=WORK_DIR/'robust_statistics/compact_semantic_review/clip_clip.py'


def main():
    path=OUTPUT/'runtime_controls.json';OUTPUT.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve base runtime controls')
    source=SOURCE.read_text();tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_transform','_convert_image_to_rgb')]
    if len(functions)!=2:raise ValueError('Pinned transform helpers changed')
    import torchvision.transforms as transforms
    namespace={name:getattr(transforms,name) for name in ('Compose','Resize','CenterCrop','ToTensor','Normalize')}
    namespace['BICUBIC']=transforms.InterpolationMode.BICUBIC
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),namespace)
    official=namespace['_transform'](224)
    cases=[]
    for h,w in ((256,256),(301,417),(417,301)):
        y,x=np.indices((h,w));rgb=np.stack([(x+y)%256,(3*x+y)%256,(x+7*y)%256],axis=2).astype(np.uint8)
        expected=official(Image.fromarray(rgb)).numpy();actual=prepare224(rgb)
        if not np.array_equal(expected,actual):raise ValueError('Base preprocessing differs from original transform')
        cases.append({'height':h,'width':w,'maximum_preprocessing_difference':0})
    encoder=FrozenClipBase(MODELS_ROOT/'clip_small/ViT-B-16.pt');torch=encoder.torch
    timings=[]
    with torch.inference_mode(),torch.jit.optimized_execution(False):
        tensor=torch.linspace(-1,1,3*224*224,device=encoder.device).reshape(1,3,224,224).half()
        for index in range(120):
            torch.cuda.synchronize();start=perf_counter();encoder.model(tensor);torch.cuda.synchronize()
            if index>=20:timings.append(1000*(perf_counter()-start))
        generator=torch.Generator(device=encoder.device).manual_seed(20261006)
        for value in (torch.zeros_like(tensor),tensor,torch.randn(tensor.shape,device=encoder.device,generator=generator).half()):
            output=encoder.model(value);repeat=encoder.model(value)
            if not torch.equal(output,repeat) or not torch.isfinite(output).all():raise ValueError('Base synthetic repeat failed')
    write_json(path,{'passed':True,'synthetic_only':True,'native_images_used':0,'preprocessing_cases':cases,
            'encoder':encoder.provenance,'forward_only_p50_ms':float(np.quantile(timings,.5)),
            'forward_only_p95_ms':float(np.quantile(timings,.95)),
            'source_transform_sha256':file_sha256(SOURCE),'script_sha256':file_sha256(Path(__file__)),
            'representation_sha256':file_sha256(REPO_ROOT/'src/palimpsest/detection/representations/frozen_clip_base.py'),
            'scope':'Original AST transform helpers executed without expression edits;synthetic CUDA forward only,not end-to-end'})
    print(json.dumps({'passed':True,'forward_only_p95_ms':float(np.quantile(timings,.95))}),flush=True)


if __name__=='__main__':main()
