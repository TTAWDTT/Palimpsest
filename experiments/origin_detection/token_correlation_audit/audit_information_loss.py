"""Known same-center/spread feature bags retain different off-diagonal signal."""

import json
from pathlib import Path

import numpy as np

from palimpsest.detection.representations.token_statistics import clipped_token_statistics
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR, REPO_ROOT


def control():
    bags = (np.array([[-1.,-1.],[-1.,-1.],[1.,1.],[1.,1.]]),
            np.array([[-1.,1.],[-1.,1.],[1.,-1.],[1.,-1.]]))
    statistics = [clipped_token_statistics(bag) for bag in bags]
    if not all(np.array_equal(statistics[0][i],statistics[1][i]) for i in (0,1)):
        raise ValueError('Known clipped-center/RMS equality failed')
    covariances = [(bag-bag.mean(axis=0)).T@(bag-bag.mean(axis=0))/len(bag) for bag in bags]
    expected = (np.array([[1.,1.],[1.,1.]]),np.array([[1.,-1.],[-1.,1.]]))
    if not all(np.array_equal(a,b) for a,b in zip(covariances,expected)):
        raise ValueError('Known covariance matrix failed')
    if np.array_equal(covariances[0],covariances[1]):
        raise ValueError('Incorrect equal covariance claim accepted')
    return {'center_equal':True,'unit_RMS_equal':True,'covariance_a':covariances[0].tolist(),
            'covariance_b':covariances[1].tolist(),'incorrect_same_covariance_claim_rejected':True,
            'scope':'Analytic finite bags;not image invariance or classifier performance'}


def main():
    output = WORK_DIR/'robust_statistics/compact_covariance_review/information_control.json'
    if output.exists():raise FileExistsError('Preserve information control')
    output.parent.mkdir(parents=True,exist_ok=True)
    value = control()
    paths = (Path(__file__),Path(__file__).parent/'README.md',
             REPO_ROOT/'src/palimpsest/detection/representations/token_statistics.py')
    value['code_pins'] = {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in paths}
    output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value))


if __name__=='__main__':main()
