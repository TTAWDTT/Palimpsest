"""Audit percentage boundaries using exact class counts; preserve prior receipts."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from palimpsest.io.hashing import file_sha256


def exact_ba(metric):
    fractions = []
    for label in ('real', 'fake'):
        count = int(metric[f'{label}_images'])
        value = float(metric[f'{label}_accuracy_at_zero']) * count
        correct = round(value)
        if count <= 0 or abs(correct-value) > 1e-8 or not 0 <= correct <= count:
            raise ValueError('Rate does not encode an integer class count')
        fractions.append(Fraction(correct, count))
    result = sum(fractions)/2
    if abs(float(result)-metric['balanced_accuracy_at_zero']) > 1e-12:
        raise ValueError('Reported BA disagrees with class counts')
    return result


def audit(receipt, limit):
    result = {}
    for candidate, value in receipt['candidates'].items():
        comparisons = []
        for variant, key, flag in (
            ('jpeg90', 'recompression_aggregate', 'recompression_ba_drop'),
            ('jpeg70', 'held_encoding_aggregate', 'held_encoding_ba_drop'),
        ):
            drops = []
            for domain, summary in value[key].items():
                for condition, metric in summary['conditions'].items():
                    if condition == 'original':
                        continue
                    before = exact_ba(value['selection_aggregate'][domain]['conditions'][condition])
                    after = exact_ba(metric)
                    drop = before-after
                    drops.append(drop)
                    comparisons.append({'variant': variant, 'domain': domain, 'condition': condition,
                                        'before': str(before), 'after': str(after), 'drop': str(drop),
                                        'drop_pp': float(100*drop), 'within_limit': drop <= limit})
            corrected = all(drop <= limit for drop in drops)
            comparisons.append({'criterion': flag, 'original': value['criteria'][flag], 'corrected': corrected})
        criteria = dict(value['criteria'])
        for item in comparisons:
            if 'criterion' in item:
                criteria[item['criterion']] = item['corrected']
        result[candidate] = {'comparisons': comparisons, 'corrected_criteria': criteria,
                             'corrected_gate_passed': all(criteria.values())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--maximum-drop', default='0.05')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve boundary audit')
    limit = Fraction(args.maximum_drop)
    if not 0 <= limit <= 1:
        raise ValueError('Drop limit outside [0,1]')
    receipt = json.loads(args.receipt.read_text(encoding='utf-8'))
    result = {'parent_sha256': file_sha256(args.receipt), 'script_sha256': file_sha256(Path(__file__)),
              'limit': str(limit), 'candidates': audit(receipt, limit),
              'scope': 'Exact reconstruction of class counts; original scores/code/receipt unchanged'}
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v['corrected_gate_passed'] for k, v in result['candidates'].items()}))


if __name__ == '__main__':
    main()
