"""Signed streaming covariance extension; exact unchanged3072 prefix gate."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.representations.frozen_cure_covariance import FrozenCureCovariance,FEATURE_NAMES
from palimpsest.evaluation.numeric_features import numeric_rows
from palimpsest.evaluation.numeric_extension import extract_extension
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR,REPO_ROOT,MODELS_ROOT
from experiments.origin_detection.cure_token_statistics.prepare_features import (
    OUTPUT as PARENT,synthetic_gate,audit_cache as parent_audit)
from experiments.origin_detection.cure_baseline.run_iteration import VENDOR,source_pins,runtime
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS,audit_receipt
from experiments.origin_detection.phase_statistics.run_iteration import image_path,write_json
from palimpsest.detection.algorithms.residual_statistics.features import resize256

OUTPUT=WORK_DIR/'robust_statistics/cure_token_covariance'


def code_pins():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    paths+=[REPO_ROOT/p for p in (
        'src/palimpsest/detection/representations/token_covariance.py',
        'src/palimpsest/detection/representations/frozen_cure_covariance.py',
        'src/palimpsest/detection/representations/frozen_cure_tokens.py',
        'src/palimpsest/evaluation/numeric_extension.py','src/palimpsest/evaluation/numeric_features.py',
        'src/palimpsest/evaluation/features.py','src/palimpsest/evaluation/source_readout_campaign.py',
        'src/palimpsest/evaluation/source_training.py','src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/detection/algorithms/source_view_risk.py','src/palimpsest/evaluation/robust_views.py',
        'src/palimpsest/detection/algorithms/paired_stability.py',
        'src/palimpsest/evaluation/pairing.py','src/palimpsest/evaluation/classification.py',
        'experiments/origin_detection/cure_token_statistics/prepare_features.py',
        'experiments/origin_detection/cure_baseline/run_iteration.py',
        'experiments/origin_detection/source_view_risk/run_iteration.py',
        'experiments/origin_detection/token_consistency_crossfit/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'tests/detection/test_token_covariance.py','tests/evaluation/test_numeric_extension.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)}


def parent_data():
    receipt=audit_receipt(PARENT/'features.json')
    if (not receipt['all_original_exact'] or receipt['records']!=15120
            or file_sha256(PARENT/'vectors.npy')!=receipt['vectors_sha256']
            or file_sha256(PARENT/'metadata.csv')!=receipt['metadata_sha256']):
        raise ValueError('Signed token parent changed')
    metadata=read_rows(PARENT/'metadata.csv');values=np.load(PARENT/'vectors.npy',mmap_mode='r')
    inventory=[r for r in metadata if r['variant']=='raw']
    parent_audit(metadata,values,inventory)
    return inventory,metadata,values,receipt


def audit_cache(metadata,values,inventory):
    rows=numeric_rows(metadata,values,FEATURE_NAMES)
    for role in ('fit','threshold','selection'):
        selected=[r for r in inventory if r['role']==role]
        if selected:validate_feature_cache([r for r in rows if r['role']==role],selected,FEATURE_NAMES,
            variants=VARIANTS if role=='selection' else VARIANTS[:2],bounds=(-np.inf,np.inf))
    covariance=values[:,3072:];norms=np.linalg.norm(covariance,axis=1)
    if np.any((norms!=0)&(np.abs(norms-1)>1e-10)) or np.any(np.abs(covariance)>1+1e-10):
        raise ValueError('Covariance descriptor norm/bounds differ')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--pilot',action='store_true');g.add_argument('--prepare',action='store_true');args=parser.parse_args()
    output=OUTPUT.with_name(OUTPUT.name+'_pilot') if args.pilot else OUTPUT
    output.mkdir(parents=True,exist_ok=True)
    if (output/'features.json').exists():raise FileExistsError('Preserve covariance receipt')
    controls=json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins']!=code_pins():raise ValueError('Covariance controls changed')
    inventory,metadata,values,parent=parent_data();packages=runtime();cv2.setNumThreads(1)
    if args.pilot:
        p=PARENT.with_name(PARENT.name+'_pilot');receipt=audit_receipt(p/'features.json')
        if file_sha256(p/'metadata.csv')!=receipt['metadata_sha256']:raise ValueError('Exposed token pilot changed')
        names={r['filename'] for r in read_rows(p/'metadata.csv')}
        inventory=[r for r in inventory if r['filename'] in names]
        if len(inventory)!=60 or any(r['role']!='selection' for r in inventory):
            raise ValueError('Expected old60exposed pilot images')
    else:
        p=OUTPUT.with_name(OUTPUT.name+'_pilot');pilot=json.loads((p/'features.json').read_text())
        if (pilot['code_pins']!=code_pins() or not pilot['cost_passed'] or pilot['runtime']!=packages
                or pilot['repeated_raw_records']!=60 or not pilot['all_parent_prefix_exact']
                or pilot['parent_features_sha256']!=file_sha256(PARENT/'features.json')
                or file_sha256(p/'vectors.npy')!=pilot['vectors_sha256']
                or file_sha256(p/'metadata.csv')!=pilot['metadata_sha256']):
            raise ValueError('Covariance pilot changed/failed')
    encoder=FrozenCureCovariance(VENDOR,VENDOR/'weights/cure_adapter.pt',
        MODELS_ROOT/'cure/PE-Core-L14-336.pt',source_pins=source_pins())
    try:
        with threadpool_limits(limits=1):
            witnesses=synthetic_gate(encoder,output) if args.pilot else pilot['synthetic_witnesses']
            result=extract_extension(inventory,metadata,values,output,FEATURE_NAMES,encoder,
                resolve_path=image_path,variants=VARIANTS,resize=resize256,
                jpeg_parameters={VARIANTS[1]:(90,0),VARIANTS[2]:(70,2),VARIANTS[3]:(60,2)},
                repeat_raw=args.pilot,validate=audit_cache)
    finally:encoder.close()
    passed=result['cost_median_ms']<=175 if args.pilot else None
    write_json(output/'features.json',{**result,'feature_names':FEATURE_NAMES,'code_pins':code_pins(),
        'runtime':packages,'encoder':encoder.provenance,'synthetic_witnesses':witnesses,'cost_passed':passed,
        'parent_features_sha256':file_sha256(PARENT/'features.json'),'inventory_sha256':parent['inventory_sha256'],
        'projection_sha256':encoder.provenance['projection_sha256'],
        'scope':'Exposed development token correlation extension;no image invariance or deployment result'})
    if args.pilot and not passed:raise ValueError('Covariance pilot engineering budget refused')


if __name__=='__main__':main()
