"""Fit-only paired deltas and weighted, explicitly regularized readout."""

import numpy as np

from palimpsest.detection.algorithms.paired_stability import fit_stable_rule
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES
from palimpsest.evaluation.features import validate_feature_cache
from experiments.origin_detection.ordinal_statistics.fit_rules import choose_threshold
from experiments.origin_detection.residual_statistics.fit_rules import all_views
from experiments.origin_detection.residual_training.fit_rules import training_arrays, TRAINING_VARIANT


def paired_deltas(rows, role, variants, names=FEATURE_NAMES):
    groups = {}
    for r in rows:
        if r['role'] == role and r['variant'] in variants:
            groups.setdefault(r['src'], []).append(r)
    if not groups: raise ValueError('Empty paired role')
    counts = {d: sum(g[0]['domain'] == d for g in groups.values()) for d in ('rr', 'chimera')}
    if min(counts.values()) == 0: raise ValueError('Paired role missing domain')
    deltas, weights = [], []
    for src, records in sorted(groups.items()):
        domain = records[0]['domain']
        conditions = ('original', 'transfer', 'redigital') if domain == 'rr' else ('original', 'mac_iphone', 'lg_blackfly')
        by_key = {(r['condition'], r['variant']): r for r in records}
        if (len(by_key) != len(records) or set(by_key) != {(c, v) for c in conditions for v in variants}
                or len({(r['label'], r['domain'], r['scene'], r['role']) for r in records}) != 1):
            raise ValueError('Conflicting/incomplete paired role')
        reference = np.array([float(by_key['original', 'raw'][n]) for n in names])
        for key, r in sorted(by_key.items()):
            if key == ('original', 'raw'): continue
            deltas.append(np.array([float(r[n]) for n in names])-reference)
            weights.append(.5/(counts[domain]*(len(records)-1)))
    return np.asarray(deltas), np.asarray(weights)


def fit_rule(rows, strength, config, manifest_sha, *, shuffled=False, dimension=None):
    inventory = [r for r in rows if r['variant'] == 'raw']
    validate_feature_cache(rows, inventory, FEATURE_NAMES, variants=('raw', TRAINING_VARIANT), bounds=(0, 1))
    x, y, weights, _ = training_arrays(rows, 'equal_domain_augmented', config['seed'], shuffled=shuffled)
    names = FEATURE_NAMES if dimension is None else FEATURE_NAMES[:dimension]
    deltas, pair_weights = paired_deltas(rows, 'fit', ('raw', TRAINING_VARIANT), names)
    rule, diagnostics = fit_stable_rule(x[:, :len(names)], y, weights, deltas, pair_weights, feature_names=names,
                                        strength=strength, ridge=config['ridge'], scale_floor=config['scale_floor'],
                                        manifest_sha=manifest_sha)
    rule = choose_threshold(rule, all_views(rows, names, 'threshold'))
    diagnostics.update({'fit_records': len(x), 'paired_records': len(deltas), 'feature_count': len(names)})
    return rule, diagnostics
