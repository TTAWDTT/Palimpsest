"""Pinned smaller semantic CLIP archive; neural frozen representation, not D3."""

from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.io.hashing import file_sha256
from .frozen_clip import prepare224,ClipFeatures

WEIGHT_SHA256='5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
FEATURE_NAMES=tuple(f'frozen_clip_base/unit_shifted/{i}' for i in range(512))


def shifted_base_unit(values):
    v=np.asarray(values,np.float64)
    if v.shape!=(512,) or not np.isfinite(v).all() or np.linalg.norm(v)<=1e-12:
        raise ValueError('Invalid base CLIP feature')
    return (v/np.linalg.norm(v)+1)/2


class FrozenClipBase:
    def __init__(self,checkpoint:Path,*,device='cuda:0'):
        if file_sha256(checkpoint)!=WEIGHT_SHA256:raise ValueError('Base CLIP weight SHA differs')
        if not str(device).startswith('cuda'):raise ValueError('Fixed base trial uses CUDA')
        import torch
        self.torch=torch;self.device=torch.device(device)
        archive=torch.jit.load(str(checkpoint),map_location='cpu').eval()
        state=archive.visual.state_dict()
        count=sum(k.endswith('.attn.in_proj_weight') for k in state)
        if (state['conv1.weight'].shape!=(768,3,16,16) or state['positional_embedding'].shape!=(197,768)
                or state['proj'].shape!=(768,512) or count!=12):
            raise ValueError('Expected official ViT-B/16 dimensions')
        self.model=archive.visual.eval().to(self.device)
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with torch.inference_mode(),torch.jit.optimized_execution(False):
            witness=torch.linspace(-1,1,3*224*224,device=self.device).reshape(1,3,224,224).half()
            actual=self.model(witness);repeat=self.model(witness)
            if actual.shape!=(1,512) or not torch.equal(actual,repeat) or not torch.isfinite(actual).all():
                raise ValueError('Base frozen archive witness failed')
        self.provenance={'weights_sha256':WEIGHT_SHA256,'device':str(self.device),'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(self.device),'precision':'official archive.visual fp16',
            'backbone_parameters_trained':0,'all_parameters_frozen':all(not p.requires_grad for p in self.model.parameters()),
            'layers':count,'patch':16,'descriptor_dimension':512,'jit_optimized_execution':False,
            'synthetic_repeat_exact':True,'input':'Same frozen_clip.prepare224;stored orientation',
            'storage':'(L2_unit512+1)/2','torch_module':str(Path(torch.__file__).resolve()),'numpy_module':str(Path(np.__file__).resolve())}
        if not self.provenance['all_parameters_frozen']:raise ValueError('Base encoder not frozen')

    def extract(self,image):
        start=perf_counter();pixels=prepare224(image);ready=perf_counter();torch=self.torch
        with torch.inference_mode(),torch.jit.optimized_execution(False):
            tensor=torch.from_numpy(pixels.copy()).unsqueeze(0).to(self.device).half()
            values=self.model(tensor)[0].float().cpu().numpy()
            torch.cuda.synchronize(self.device)
        return ClipFeatures(shifted_base_unit(values),1000*(ready-start),1000*(perf_counter()-ready))
