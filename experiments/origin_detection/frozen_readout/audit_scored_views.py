"""Independent integer confusion-count audit of saved selection margins.

No fitting, model inference, production evaluator or bootstrap is called. This
checks arithmetic transcription, not source independence or physical validity.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

from palimpsest.io.hashing import file_sha256


def audit(scores,candidates):
    output={}
    for candidate,rows in scores.items():
        groups=defaultdict(list);seen=set()
        for r in rows:
            key=tuple(r[k] for k in ('domain','scene','condition','variant','src'))
            if key in seen or r['role']!='selection':raise ValueError('Duplicate/non-selection saved score')
            seen.add(key);groups[key[:4]].append(r)
            if key[1]!='all':groups[(key[0],'all',*key[2:4])].append(r)
        metrics={}
        for key,records in groups.items():
            counts={label:(sum((r['score']>0)==(label=='FAKE') for r in records if r['label']==label),
                           sum(r['label']==label for r in records)) for label in ('REAL','FAKE')}
            if any(n==0 for _,n in counts.values()):raise ValueError('Missing score class')
            value=sum(Fraction(correct,n) for correct,n in counts.values())/2
            expected=candidates[candidate]['metrics']['/'.join(key)]
            if (abs(float(value)-expected['balanced_accuracy_at_zero'])>1e-15
                    or expected['real_images']!=counts['REAL'][1] or expected['fake_images']!=counts['FAKE'][1]):
                raise ValueError('Saved confusion count differs from receipt')
            metrics['/'.join(key)]={'class_integer_counts':counts,'exact_ba':str(value)}
        for name,pair in candidates[candidate]['pairs'].items():
            domain,scene,condition,variant=name.split('/')
            if '>' in condition:
                before,after=condition.split('>');keys=((domain,scene,before,variant),(domain,scene,after,variant))
            else:
                before,after=variant.split('>');keys=((domain,scene,condition,before),(domain,scene,condition,after))
            first,second=({r['src']:r for r in groups[k]} for k in keys)
            if set(first)!=set(second):raise ValueError('Saved pair coverage differs')
            drop=Fraction(metrics['/'.join(keys[0])]['exact_ba'])-Fraction(metrics['/'.join(keys[1])]['exact_ba'])
            if str(drop)!=pair['exact_ba_drop']:raise ValueError('Saved pair exact rate differs')
        output[candidate]={'metric_groups':len(metrics),'paired_comparisons':len(candidates[candidate]['pairs']),
                           'metrics':metrics,'passed':True}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    path=args.directory/'score_counts_audit.json'
    if path.exists():raise FileExistsError('Preserve score-count audit')
    receipt=json.loads((args.directory/'iteration.json').read_text(encoding='utf-8'))
    score_path=args.directory/'selection_scores.json'
    if file_sha256(score_path)!=receipt['selection_scores_sha256']:raise ValueError('Saved margins changed')
    scores=json.loads(score_path.read_text(encoding='utf-8'))
    result=audit(scores,receipt['candidates'])
    # Deliberately corrupt one BA receipt; the same audit must reject it.
    bad=json.loads(json.dumps(receipt['candidates']))
    candidate=next(iter(bad));view=next(iter(bad[candidate]['metrics']))
    bad[candidate]['metrics'][view]['balanced_accuracy_at_zero']+=.01
    rejected=False
    try:audit(scores,bad)
    except ValueError as error:
        if 'confusion count' not in str(error):raise
        rejected=True
    if not rejected:raise ValueError('Confusion-count negative control did not fail')
    value={'passed':True,'candidates':result,'perturbed_ba_rejected':rejected,
           'iteration_sha256':file_sha256(args.directory/'iteration.json'),'scores_sha256':file_sha256(score_path),
           'script_sha256':file_sha256(Path(__file__)),
           'scope':'Integer rate arithmetic only,shared model scores;not independent scientific validation'}
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'candidates':len(result),'perturbed_ba_rejected':True}))


if __name__=='__main__':main()
