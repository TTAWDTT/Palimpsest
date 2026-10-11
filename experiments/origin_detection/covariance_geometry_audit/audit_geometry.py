"""Known SPD-distance identity does not certify fixed classifier invariance."""

from decimal import Decimal,localcontext
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR,REPO_ROOT


def distance(a,b):
    values=eigvalsh(b,a)
    if np.any(values<=0):raise ValueError('SPD matrices required')
    return float(np.linalg.norm(np.log(values)))


def main():
    path=WORK_DIR/'robust_statistics/riemannian_covariance_review/geometry_control.json'
    if path.exists():raise FileExistsError('Preserve geometry control')
    path.parent.mkdir(parents=True,exist_ok=True)
    real=(Fraction(1,2),Fraction(1,2));fake=(Fraction(4,5),Fraction(1,5))
    ratios=tuple(b/a for a,b in zip(real,fake))
    if ratios!=(Fraction(8,5),Fraction(2,5)):raise ValueError('Known ratios failed')
    doubled=tuple(a*r*r for a,r in zip(real,ratios))
    if tuple(b/a for a,b in zip(fake,doubled))!=ratios:raise ValueError('Repeated transform ratios failed')
    values={}
    for digits in (50,100):
        with localcontext() as context:
            context.prec=digits
            logs=[(Decimal(r.numerator)/Decimal(r.denominator)).ln() for r in ratios]
            values[str(digits)]=str(sum(x*x for x in logs).sqrt())
    if abs(Decimal(values['50'])-Decimal(values['100']))>Decimal('1e-48'):
        raise ValueError('Raised-precision diagonal distance changed')
    matrices=[np.diag([float(v) for v in s]) for s in (real,fake,doubled)]
    d=distance(matrices[0],matrices[1])
    if abs(d-float(values['100']))>1e-14 or abs(distance(matrices[0],matrices[2])-2*d)>1e-14:
        raise ValueError('Generalized eigenvalue versus diagonal formula differs')
    common=np.diag([3.,2.])
    transformed=[common@m@common.T for m in matrices[:2]]
    if abs(distance(*transformed)-d)>1e-14:raise ValueError('Both-reference congruence control failed')
    prediction=lambda q:int(distance(matrices[0],q)>distance(matrices[1],q))
    before=[prediction(m) for m in matrices[:2]];after=[prediction(m) for m in matrices[1:]]
    if before!=[0,1] or after!=[1,1]:raise ValueError('Known fixed-reference decision flip failed')
    wrong_accepted=before==after
    if wrong_accepted:raise ValueError('Incorrect unchanged decisions accepted')
    pins=[Path(__file__),Path(__file__).parent/'README.md']
    output={'diagonal_distance_decimal':values,'float_distance':d,'common_transform_distance_preserved':True,
        'predictions_before':before,'predictions_after':after,'known_ba_before':'1','known_ba_after':'1/2',
        'wrong_unchanged_decision_claim_rejected':True,'code_pins':{str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in pins},
        'scope':'Selected analytic SPD matrices;shared chosen inputs,different arithmetic paths;not physical recapture or independent scientific validation'}
    path.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8');print(json.dumps(output))


if __name__=='__main__':main()
