"""A global threshold selected from calibration-role matched processing views.

Exact integer BA/drop comparisons use an LCM denominator. Calibration fit is
not a guarantee for later source/condition samples and is not an oracle test.
"""

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from math import lcm

import numpy as np


def paired_threshold(rule, views, *, variants, minimum_ba=.8, maximum_drop=.02):
    if not views or not variants or len(set(variants)) != len(variants):
        raise ValueError('Missing calibration views/variant order')
    records, scores, labels, seen = [], [], [], set()
    for rows, values, truth in views.values():
        actual = np.asarray(rule.score(values), float)
        truth = np.asarray(truth)
        if len(rows) != len(actual) or truth.shape != actual.shape or not np.isfinite(actual).all():
            raise ValueError('Calibration score dimensions/finite values')
        for row, score, label in zip(rows, actual, truth):
            key = tuple(row[k] for k in ('domain', 'scene', 'condition', 'variant', 'src'))
            if row['role'] != 'threshold' or key in seen or label not in (0, 1) or row['variant'] not in variants:
                raise ValueError('Duplicate/unregistered/non-calibration input')
            seen.add(key); records.append(row); scores.append(float(score)); labels.append(int(label))
    grouped = defaultdict(list)
    for i, row in enumerate(records):
        key = tuple(row[k] for k in ('domain', 'scene', 'condition', 'variant'))
        grouped[key].append(i)
        if row['scene'] != 'all':
            grouped[(key[0], 'all', *key[2:])].append(i)
    keys = sorted(grouped); indices = {key: i for i, key in enumerate(keys)}
    counts = np.array([[sum(labels[i] == label for i in grouped[key]) for label in (0, 1)] for key in keys], int)
    if np.any(counts == 0):
        raise ValueError('Calibration class absent')
    denominator = lcm(*(2*int(n) for n in counts.flat))
    factors = np.array([[denominator//(2*int(n)) for n in ns] for ns in counts], dtype=np.int64)
    # Before minimum score every query predicts FAKE.
    correct = np.c_[np.zeros(len(keys), dtype=np.int64), counts[:, 1].copy()]
    membership = [[] for _ in records]
    for key, rows in grouped.items():
        for row in rows:
            membership[row].append(indices[key])
    pairs = []
    def add(before, after):
        first = {records[i]['src']: labels[i] for i in grouped[before]}
        second = {records[i]['src']: labels[i] for i in grouped[after]}
        if first != second:
            raise ValueError('Incomplete/conflicting matched calibration sources')
        pairs.append((indices[before], indices[after]))
    for key in keys:
        domain, scene, condition, variant = key
        if condition != 'original':
            before = (domain, scene, 'original', variant)
            if before not in grouped:
                raise ValueError('Missing calibration original')
            add(before, key)
    for domain, scene, condition in sorted({k[:3] for k in keys}):
        if {(domain, scene, condition, v) for v in variants} - set(grouped):
            raise ValueError('Incomplete calibration encodings')
        for before, after in combinations(variants, 2):
            add((domain, scene, condition, before), (domain, scene, condition, after))
    if not pairs:
        raise ValueError('Calibration has no robustness pairs')
    first = np.array([a for a, _ in pairs]); second = np.array([b for _, b in pairs])
    ba_floor, drop_cap = Fraction(str(minimum_ba)), Fraction(str(maximum_drop))
    best = None; states = 0
    def consider(threshold):
        nonlocal best, states
        units = (correct*factors).sum(axis=1)
        worst_ba = int(units.min()); worst_drop = int((units[first]-units[second]).max())
        ba_ok = worst_ba*ba_floor.denominator >= ba_floor.numerator*denominator
        drop_ok = worst_drop*drop_cap.denominator <= drop_cap.numerator*denominator
        # All classes share denominator/2 because BA factors include the half.
        worst_class_units = int((correct*factors*2).min())
        if ba_ok and drop_ok:
            rank = (2, worst_ba, -worst_drop, worst_class_units, -abs(threshold), -threshold)
        elif ba_ok:
            rank = (1, -worst_drop, worst_ba, worst_class_units, -abs(threshold), -threshold)
        else:
            rank = (0, worst_ba, -worst_drop, worst_class_units, -abs(threshold), -threshold)
        if best is None or rank > best[0]:
            best = rank, threshold, worst_ba, worst_drop, worst_class_units
        states += 1
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    consider(float(np.nextafter(scores[order[0]], -np.inf)))
    cursor = 0
    while cursor < len(order):
        threshold = scores[order[cursor]]
        end = cursor
        while end < len(order) and scores[order[end]] == threshold:
            row = order[end]; label = labels[row]
            for group in membership[row]:
                correct[group, label] += 1 if label == 0 else -1
            end += 1
        consider(threshold); cursor = end
    _, threshold, ba, drop, class_units = best
    return replace(rule, threshold=threshold), {'policy': 'calibration matched BA/drop feasibility then fixed fallback',
        'calibration_records': len(records), 'groups': len(keys), 'paired_comparisons': len(pairs),
        'threshold_states': states, 'exact_denominator': denominator,
        'minimum_exact_calibration_ba': str(Fraction(ba, denominator)),
        'maximum_exact_calibration_drop': str(Fraction(drop, denominator)),
        'minimum_exact_calibration_class_accuracy': str(Fraction(class_units, denominator)),
        'joint_calibration_feasible': best[0][0] == 2,
        'scope': 'Current threshold-role pairs only;not outer acceptance or image-channel invariance'}
