"""Exact rank/tie counts and source-fold coverage of a fit-only diagnostic."""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT


def exact_auc(rows):
    negatives = sorted(r['score'] for r in rows if r['label'] == 'REAL')
    positives = [r['score'] for r in rows if r['label'] == 'FAKE']
    if not negatives or not positives:
        raise ValueError('Absent ranking class')
    twice_wins = sum(bisect_left(negatives, s)+bisect_right(negatives, s) for s in positives)
    return Fraction(twice_wins, 2*len(negatives)*len(positives))


def verify(rows, metrics, folds):
    grouped, seen = defaultdict(list), set()
    source_folds = defaultdict(set)
    for r in rows:
        key = tuple(r[k] for k in ('domain', 'scene', 'condition', 'src'))
        if (key in seen or r['role'] != 'fit' or r['variant'] != 'raw'
                or r['label'] not in ('REAL', 'FAKE') or not math.isfinite(r['score'])):
            raise ValueError('Invalid fit-only ranking row')
        seen.add(key)
        source = r['domain']+'/'+r['src']
        source_folds[source].add(r['fold'])
        if r['fold'] != folds[source]:
            raise ValueError('Wrong recorded source fold')
        grouped['/'.join(key[:3])].append(r)
        if r['scene'] != 'all':
            grouped[r['domain']+'/all/'+r['condition']].append(r)
    if len(rows) != 3780 or len(source_folds) != 1260 or set(source_folds) != set(folds):
        raise ValueError('Missing diagnostic source/rows')
    if any(len(v) != 1 for v in source_folds.values()) or len(grouped) != 15 or set(grouped) != set(metrics):
        raise ValueError('Diagnostic grouping changed')
    result = {}
    for name, group in grouped.items():
        value = exact_auc(group)
        quoted = metrics[name]
        if (abs(float(value)-quoted['auc']) > 1e-15 or len(group) != quoted['rows']
                or sum(r['label'] == 'FAKE' for r in group) != quoted['fake_rows']):
            raise ValueError('Ranking arithmetic changed')
        per_fold = {str(i): str(exact_auc([r for r in group if r['fold'] == i])) for i in range(5)}
        result[name] = {'pooled_exact_auc': str(value), 'per_fold_exact_auc': per_fold,
            'unweighted_fold_mean_auc': float(sum(map(Fraction, per_fold.values()))/5)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    out = args.directory/'ranking_audit.json'
    if out.exists():
        raise FileExistsError('Preserve ranking audit')
    receipt = json.loads((args.directory/'diagnostic.json').read_text())
    scores_path = args.directory/'oof_scores.json'
    if file_sha256(scores_path) != receipt['oof_sha256']:
        raise ValueError('Diagnostic scores changed')
    for name, expected in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/name) != expected:
            raise ValueError('Diagnostic source changed')
    toy = [{'score': 0., 'label': 'REAL'}, {'score': 0., 'label': 'FAKE'},
           {'score': 1., 'label': 'FAKE'}]
    if exact_auc(toy) != Fraction(3, 4):
        raise ValueError('Known tied ranking failed')
    scores = json.loads(scores_path.read_text())
    if set(scores) != set(receipt['metrics']) or len(scores) != 6:
        raise ValueError('Model/label modes changed')
    results = {key: verify(rows, receipt['metrics'][key], receipt['fold_sources'])
               for key, rows in scores.items()}
    key = next(iter(scores))
    bad = deepcopy(receipt['metrics'][key])
    bad[next(iter(bad))]['auc'] += .01
    try:
        verify(scores[key], bad, receipt['fold_sources'])
    except ValueError:
        pass
    else:
        raise ValueError('Wrong AUC accepted')
    bad_rows = deepcopy(scores[key]); bad_rows[0]['fold'] = (bad_rows[0]['fold']+1)%5
    try:
        verify(bad_rows, receipt['metrics'][key], receipt['fold_sources'])
    except ValueError:
        pass
    else:
        raise ValueError('Wrong source fold accepted')
    out.write_text(json.dumps({'passed': True, 'results': results, 'wrong_auc_rejected': True,
        'wrong_fold_rejected': True, 'scores_sha256': file_sha256(scores_path),
        'script_sha256': file_sha256(Path(__file__)), 'scope': 'Saved ranks/role records only;no optimizer retraining or independence certification'},
        indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
