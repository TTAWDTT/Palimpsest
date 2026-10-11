"""Post-exposure common-threshold feasibility; never save an adopted rule.

Reuses the exact tied-event sweep. Parent receipts authenticate saved scores,
not a replay of historical producers or a test of independent images.
"""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
from time import perf_counter

from experiments.origin_detection.frozen_readout.audit_scored_views import audit
from experiments.origin_detection.score_feasibility.audit_thresholds import groups_and_pairs, sweep
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT


def flip_details(rows, groups, pairs):
    """All maximum-drop ties at the saved zero threshold, with source flips."""
    values = []
    for before, after in pairs:
        first = {rows[i]['src']: rows[i] for i in groups[before]}
        second = {rows[i]['src']: rows[i] for i in groups[after]}
        if first.keys() != second.keys():
            raise ValueError('Inconsistent flip source coverage')
        counts, flips = Counter(), []
        denominators = Counter(r['label'] for r in first.values())
        if set(denominators) != {'REAL', 'FAKE'}:
            raise ValueError('Missing flip class')
        for source in sorted(first):
            a, b = first[source], second[source]
            if a['label'] != b['label']:
                raise ValueError('Conflicting flip label')
            signed = 1 if a['label'] == 'FAKE' else -1
            correct_a = (a['score'] > 0) == (signed == 1)
            correct_b = (b['score'] > 0) == (signed == 1)
            counts[a['label']] += int(correct_a)-int(correct_b)
            if correct_a != correct_b:
                flips.append({'src': source, 'label': a['label'],
                    'transition': 'correct_to_wrong' if correct_a else 'wrong_to_correct',
                    'before_margin': a['score'], 'after_margin': b['score'],
                    'signed_degradation': signed*(a['score']-b['score'])})
        drop = sum(Fraction(counts[label], denominators[label]) for label in denominators)/2
        values.append({'before': before, 'after': after, 'exact_drop': str(drop), 'flips': flips})
    maximum = max(Fraction(v['exact_drop']) for v in values)
    return [v for v in values if Fraction(v['exact_drop']) == maximum]


def diagnose(directory, candidate):
    receipt_path, scores_path = directory/'iteration.json', directory/'selection_scores.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if file_sha256(scores_path) != receipt['selection_scores_sha256']:
        raise ValueError('Signed saved score bytes changed')
    rows = json.loads(scores_path.read_text(encoding='utf-8'))[candidate]
    expected = receipt['candidates'][candidate]
    audit({candidate: rows}, {candidate: expected})
    groups, pairs = groups_and_pairs(rows, expected)
    result = sweep(rows, groups, pairs)
    if (len(rows) != 5040 or len(groups) != 60 or len(pairs) != 130
            or abs(float(Fraction(result['zero_threshold']['minimum_ba']))
                   - min(expected['minimum_domain_ba'], expected['minimum_scene_ba'])) > 1e-15
            or abs(float(Fraction(result['zero_threshold']['maximum_drop']))
                   - expected['maximum_any_scene_drop']) > 1e-15):
        raise ValueError('Saved panel/zero threshold mismatch')
    return {'candidate': candidate, **result, 'maximum_drop_ties': flip_details(rows, groups, pairs),
        'iteration_sha256': file_sha256(receipt_path), 'scores_sha256': file_sha256(scores_path),
        'historical_producer_code_pins': receipt['code_pins']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--candidate', default='selected')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve diagnostic receipt')
    start = perf_counter()
    result = diagnose(args.directory, args.candidate)
    result['elapsed_seconds'] = perf_counter()-start
    paths = [Path(__file__), REPO_ROOT/'tools/README.md',
        REPO_ROOT/'experiments/origin_detection/score_feasibility/audit_thresholds.py',
        REPO_ROOT/'experiments/origin_detection/frozen_readout/audit_scored_views.py',
        REPO_ROOT/'tests/evaluation/test_score_feasibility.py',
        REPO_ROOT/'tests/evaluation/test_saved_threshold_diagnostic.py']
    result['diagnostic_code_pins'] = {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths}
    result['scope'] = ('Post-exposure finite common-threshold diagnostic on signed saved scores; '
        'no adopted threshold, historical optimizer replay, new pixels or independent validation')
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('event_states', 'feasible_event_states',
        'minimum_maximum_drop_under_ba80', 'maximum_minimum_ba_under_drop2pp', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
