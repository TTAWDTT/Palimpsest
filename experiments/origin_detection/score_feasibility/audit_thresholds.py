"""Exact finite common-threshold feasibility, never a deployed calibration."""

from collections import defaultdict
from fractions import Fraction
import json
from math import isfinite, lcm
from pathlib import Path

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.frozen_readout.audit_scored_views import audit

OUTPUT = WORK_DIR / 'robust_statistics/score_feasibility'
TARGETS = (('source_view_risk', 'weak/source'), ('complementary_encoders', 'joint/source'),
           ('cure_readout', 'full/source'), ('cure_angular_risk', 'direction/source'),
           ('clip_cure_complement', 'source'))


def sweep(rows, groups, pairs):
    """Sweep tied events with integer BA increments and exact comparisons."""
    membership = defaultdict(list)
    sizes = []
    for index, (_, records) in enumerate(groups.items()):
        if len(records) != len(set(records)):
            raise ValueError('Duplicate rate-group member')
        real = sum(rows[i]['label'] == 'REAL' for i in records)
        fake = sum(rows[i]['label'] == 'FAKE' for i in records)
        if real == 0 or real != fake:
            raise ValueError('Equal positive class counts required')
        sizes.append(real)
        for i in records:
            membership[i].append(index)
    if set(membership) != set(range(len(rows))):
        raise ValueError('Incomplete score-group coverage')
    if any(not isfinite(float(r['score'])) or r['label'] not in {'REAL', 'FAKE'} for r in rows):
        raise ValueError('Invalid saved score or label')
    names = list(groups)
    indexes = {name: i for i, name in enumerate(names)}
    pair_indexes = []
    for first, second in pairs:
        if first not in groups or second not in groups:
            raise ValueError('Unknown paired group')
        a = {(rows[i]['domain'], rows[i]['src']): rows[i]['label'] for i in groups[first]}
        b = {(rows[i]['domain'], rows[i]['src']): rows[i]['label'] for i in groups[second]}
        if len(a) != len(groups[first]) or len(b) != len(groups[second]) or a != b:
            raise ValueError('Paired source coverage or labels differ')
        pair_indexes.append((indexes[first], indexes[second]))
    if not pair_indexes:
        raise ValueError('No paired groups')
    denominator = lcm(*(2 * size for size in sizes))
    numerators = [denominator // 2] * len(groups)  # initial all-FAKE
    steps = [denominator // (2 * size) for size in sizes]
    events = defaultdict(list)
    for i, row in enumerate(rows):
        events[float(row['score'])].append(i)
    feasible = 0
    minimum_drop = None
    maximum_ba = None
    zero_state = None

    def state():
        nonlocal feasible, minimum_drop, maximum_ba
        low = min(numerators)
        decline = max(numerators[a] - numerators[b] for a, b in pair_indexes)
        absolute = low * 5 >= denominator * 4
        stable = decline * 50 <= denominator
        feasible += int(absolute and stable)
        if absolute:
            minimum_drop = decline if minimum_drop is None else min(minimum_drop, decline)
        if stable:
            maximum_ba = low if maximum_ba is None else max(maximum_ba, low)
        return {'minimum_ba': str(Fraction(low, denominator)),
                'maximum_drop': str(Fraction(decline, denominator))}

    initial = state()
    for score, records in sorted(events.items()):
        if score > 0 and zero_state is None:
            zero_state = initial
        for i in records:
            direction = 1 if rows[i]['label'] == 'REAL' else -1
            for group_index in membership[i]:
                numerators[group_index] += direction * steps[group_index]
        initial = state()
    if zero_state is None:
        zero_state = initial
    return {'event_states': len(events) + 1, 'rate_groups': len(groups), 'paired_comparisons': len(pairs),
            'feasible_event_states': feasible, 'zero_threshold': zero_state,
            'minimum_maximum_drop_under_ba80': None if minimum_drop is None else str(Fraction(minimum_drop, denominator)),
            'maximum_minimum_ba_under_drop2pp': None if maximum_ba is None else str(Fraction(maximum_ba, denominator)),
            'integer_ba_denominator': denominator}


def groups_and_pairs(rows, expected):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        key = tuple(row[k] for k in ('domain', 'scene', 'condition', 'variant'))
        groups['/'.join(key)].append(i)
        if key[1] != 'all':
            groups['/'.join((key[0], 'all', *key[2:]))].append(i)
    if set(groups) != set(expected['metrics']):
        raise ValueError('Saved rate groups changed')
    pairs = []
    for name in expected['pairs']:
        domain, scene, condition, variant = name.split('/')
        if '>' in condition:
            before, after = condition.split('>')
            pair = (f'{domain}/{scene}/{before}/{variant}', f'{domain}/{scene}/{after}/{variant}')
        else:
            before, after = variant.split('>')
            pair = (f'{domain}/{scene}/{condition}/{before}', f'{domain}/{scene}/{condition}/{after}')
        pairs.append(pair)
    return dict(groups), pairs


def code_pins():
    files = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md',
        REPO_ROOT / 'tests/evaluation/test_score_feasibility.py',
        REPO_ROOT / 'experiments/origin_detection/frozen_readout/audit_scored_views.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def main():
    if (OUTPUT / 'audit.json').exists():
        raise FileExistsError('Preserve threshold diagnostic')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text(encoding='utf-8'))
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Threshold artificial controls changed')
    results, parents = {}, {}
    for directory, candidate in TARGETS:
        root = WORK_DIR / 'robust_statistics' / directory
        receipt = json.loads((root / 'iteration.json').read_text(encoding='utf-8'))
        for name, digest in receipt['code_pins'].items():
            if file_sha256(REPO_ROOT / name) != digest:
                raise ValueError('Parent code pin changed: ' + name)
        if file_sha256(root / 'selection_scores.json') != receipt['selection_scores_sha256']:
            raise ValueError('Signed score bytes changed')
        scores = json.loads((root / 'selection_scores.json').read_text(encoding='utf-8'))
        selected, expected = scores[candidate], receipt['candidates'][candidate]
        audit({candidate: selected}, {candidate: expected})
        groups, pairs = groups_and_pairs(selected, expected)
        value = sweep(selected, groups, pairs)
        if (value['rate_groups'] != 60 or value['paired_comparisons'] != 130
                or abs(float(Fraction(value['zero_threshold']['minimum_ba']))
                       - min(expected['minimum_domain_ba'], expected['minimum_scene_ba'])) > 1e-15
                or abs(float(Fraction(value['zero_threshold']['maximum_drop']))
                       - expected['maximum_any_scene_drop']) > 1e-15):
            raise ValueError('Threshold-zero identity failed')
        results[directory + '/' + candidate] = value
        parents[directory] = {'iteration_sha256': file_sha256(root / 'iteration.json'),
                             'scores_sha256': receipt['selection_scores_sha256']}
        print(json.dumps({'candidate': directory + '/' + candidate, **value}), flush=True)
    (OUTPUT / 'audit.json').write_text(json.dumps({'diagnostics': results, 'parents': parents,
        'code_pins': code_pins(), 'goal_achieved': False,
        'scope': 'Oracle common-threshold diagnostic on exposed scores;no adopted rule or independent validation'},
        indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
