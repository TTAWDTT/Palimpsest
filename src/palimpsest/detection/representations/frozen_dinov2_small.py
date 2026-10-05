"""Pinned official DINOv2-S/14; frozen CLS and pooled patch descriptors.

A neural representation diagnostic, not a newly trained encoder or a physical
invariance claim. The external official source and weights must be supplied.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.contracts import validate_rgb
from palimpsest.io.hashing import file_sha256

CLS_NAMES=tuple(f'frozen_dinov2_small/cls_unit/{i}' for i in range(384))
PATCH_NAMES=tuple(f'frozen_dinov2_small/mean_patch_unit/{i}' for i in range(384))
FEATURE_NAMES=CLS_NAMES+PATCH_NAMES
WEIGHT_SHA256='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
CODE_COMMIT='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
MEAN=np.array([.485,.456,.406],np.float32)[:,None,None]
STD=np.array([.229,.224,.225],np.float32)[:,None,None]


def prepare224(image):
    validate_rgb(image)
    h,w=image.shape[:2]
    size=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
    im=Image.fromarray(image).resize(size,Image.Resampling.BICUBIC)
    left,top=round((size[0]-224)/2),round((size[1]-224)/2)
    rgb=np.asarray(im.crop((left,top,left+224,top+224)))
    return (rgb.transpose(2,0,1).astype(np.float32)/255-MEAN)/STD


def shifted_units(cls,patches):
    vectors=(np.asarray(cls,np.float64),np.asarray(patches,np.float64))
    if any(v.shape!=(384,) or not np.isfinite(v).all() or np.linalg.norm(v)<=1e-12 for v in vectors):
        raise ValueError('Invalid small-encoder descriptor')
    return np.concatenate([(v/np.linalg.norm(v)+1)/2 for v in vectors])


@dataclass(frozen=True)
class SmallFeatures:
    values: np.ndarray
    preprocess_ms: float
    statistics_ms: float


class FrozenDinoV2Small:
    def __init__(self,checkpoint:Path,source_receipt:Path,*,device='cuda:0'):
        receipt=json.loads(source_receipt.read_text(encoding='utf-8'))
        if file_sha256(checkpoint)!=WEIGHT_SHA256 or receipt['code_commit']!=CODE_COMMIT:
            raise ValueError('Small encoder weight/code pin differs')
        vendor=Path(receipt['vendor_root'])
        if any(file_sha256(vendor/r['name'])!=r['sha256'] for r in receipt['files']):
            raise ValueError('Official small encoder source changed')
        if not str(device).startswith('cuda'): raise ValueError('Fixed small diagnostic uses CUDA')
        self.vendor=vendor
        sys.path.insert(0,str(vendor))
        os.environ['XFORMERS_DISABLED']='1'
        import torch
        import dinov2.hub.backbones as backbones
        if not Path(backbones.__file__).resolve().is_relative_to(vendor.resolve()):
            raise ValueError('Another DINOv2 package was imported')
        self.torch=torch;self.device=torch.device(device)
        self.model=backbones.dinov2_vits14(pretrained=False).eval()
        incompatible=self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys: raise ValueError('Weight coverage differs')
        self.model=self.model.to(self.device)
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        with torch.inference_mode():
            witness=torch.linspace(-1,1,3*224*224,device=self.device).reshape(1,3,224,224)
            first=self.model.forward_features(witness);second=self.model.forward_features(witness)
            for key in ('x_norm_clstoken','x_norm_patchtokens'):
                if not torch.equal(first[key],second[key]) or not torch.isfinite(first[key]).all():
                    raise ValueError('Small encoder synthetic repeat failed')
            if first['x_norm_clstoken'].shape!=(1,384) or first['x_norm_patchtokens'].shape!=(1,256,384):
                raise ValueError('Small encoder output dimensions differ')
        self.provenance={'weights_sha256':WEIGHT_SHA256,'code_commit':CODE_COMMIT,'device':str(self.device),
                         'source_receipt_sha256':file_sha256(source_receipt),'torch':torch.__version__,
                         'gpu':torch.cuda.get_device_name(self.device),'precision':'float32 TF32 disabled',
                         'all_parameters_frozen':all(not p.requires_grad for p in self.model.parameters()),
                         'backbone_parameters_trained':0,'strict_state_coverage':True,'synthetic_repeat_exact':True,'xformers_disabled':True,
                         'descriptor':'independently L2-unit CLS and mean(normalized patch tokens),affine storage',
                         'input':'PIL BICUBIC short-side256 and center224;ImageNet normalization;stored orientation'}

    def extract(self,image):
        start=perf_counter();pixels=prepare224(image);ready=perf_counter();torch=self.torch
        with torch.inference_mode():
            values=self.model.forward_features(torch.from_numpy(pixels.copy()).unsqueeze(0).to(self.device))
            cls=values['x_norm_clstoken'][0].cpu().numpy()
            mean=values['x_norm_patchtokens'][0].mean(dim=0).cpu().numpy()
            torch.cuda.synchronize(self.device)
        return SmallFeatures(shifted_units(cls,mean),1000*(ready-start),1000*(perf_counter()-ready))
