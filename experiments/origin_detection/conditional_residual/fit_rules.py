"""Cross-fitted nuisance gate and conditional paired readout, fit sources only."""

from hashlib import sha256

import numpy as np

from palimpsest.detection.algorithms.conditional_residual import (
    ConditionalRule, basis_names, conditional_basis, gate_values,
)
from palimpsest.detection.algorithms.readouts.stable_rule import fit_stable_rule
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES
from palimpsest.evaluation.features import feature_views
from experiments.origin_detection.paired_stability.fit_rules import paired_deltas
from experiments.origin_detection.residual_training.fit_rules import training_arrays, TRAINING_VARIANT
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold


def source_fold(source, folds):
    return int.from_bytes(sha256(source.encode()).digest()[:8], 'big') % folds


def fit_gate(x, identities, weights, config, manifest_sha):
    labels = np.array([variant == TRAINING_VARIANT for _, _, variant in identities], int)
    gate, diagnostics = fit_stable_rule(
        x, labels, weights, np.zeros((1, x.shape[1])), [1.], feature_names=FEATURE_NAMES,
        strength=0., ridge=config['gate_ridge'], scale_floor=config['scale_floor'], manifest_sha=manifest_sha)
    return gate, diagnostics


def prepare_gate(rows, config, manifest_sha):
    x, _, weights, identities = training_arrays(rows, 'equal_domain_augmented', config['seed'])
    fold = np.array([source_fold(s, config['gate_folds']) for s, _, _ in identities])
    g = np.full(len(x), np.nan)
    diagnostics = []
    for f in range(config['gate_folds']):
        train, held = fold != f, fold == f
        if not train.any() or not held.any():
            raise ValueError('Empty gate source fold')
        gate, diag = fit_gate(x[train], [ids for ids, ok in zip(identities, train) if ok],
                              weights[train], config, manifest_sha)
        g[held] = gate_values(gate, x[held])
        diagnostics.append({'fold': f, 'train_records': int(train.sum()), 'held_records': int(held.sum()), **diag})
    if not np.isfinite(g).all():
        raise ValueError('Uncovered cross-fitted gate')
    final, final_diag = fit_gate(x, identities, weights, config, manifest_sha)
    # Existing paired-data auditor supplies identity/completeness validation first.
    paired_deltas(rows, 'fit', ('raw', TRAINING_VARIANT))
    projected = conditional_basis(x, g)
    by_key = {}
    fit_rows = {(r['src'], r['domain'], r['variant'], r['condition']): r for r in rows if r['role'] == 'fit'}
    ordered = []
    # Reuse the parent's view traversal, then verify every feature and identity.
    for variant in ('raw', TRAINING_VARIANT):
        views = {**feature_views(rows, FEATURE_NAMES, 'fit', processed=False, variant=variant),
                 **feature_views(rows, FEATURE_NAMES, 'fit', processed=True, variant=variant)}
        ordered.extend(r for records, _, _ in views.values() for r in records)
    if len(ordered) != len(x) or len(fit_rows) != len(x):
        raise ValueError('Conditional fit coverage differs')
    for index, (r, ident) in enumerate(zip(ordered, identities)):
        if (r['src'], r['domain'], r['variant']) != ident or not np.array_equal(
                x[index], [float(r[n]) for n in FEATURE_NAMES]):
            raise ValueError('Conditional fit order differs')
        by_key[(r['src'], r['condition'], r['variant'])] = projected[index]
    grouped = {}
    for r in ordered:
        grouped.setdefault(r['src'], []).append(r)
    counts = {d: sum(z[0]['domain'] == d for z in grouped.values()) for d in ('rr', 'chimera')}
    delta, pw = [], []
    for src, records in sorted(grouped.items()):
        anchor = by_key[src, 'original', 'raw']
        for r in records:
            if (r['condition'], r['variant']) == ('original', 'raw'):
                continue
            delta.append(by_key[src, r['condition'], r['variant']]-anchor)
            pw.append(.5/(counts[r['domain']]*(len(records)-1)))
    labels = np.array([v == TRAINING_VARIANT for _, _, v in identities], int)
    return final, projected, np.asarray(delta), np.asarray(pw), {
        'crossfit_folds': diagnostics, 'final_fit': final_diag,
        'crossfit_gate_accuracy': float(((g > .5) == labels).mean()),
        'final_crossfit_gate_mae': float(np.mean(np.abs(gate_values(final, x)-g))),
        'fit_records': len(x), 'fit_sources': len(grouped), 'paired_records': len(delta)}, g


def fit_rule(rows, prepared, strength, config, manifest_sha, *, shuffled=False):
    gate, projected, delta, pw, gate_diag, _ = prepared
    _, labels, weights, identities = training_arrays(
        rows, 'equal_domain_augmented', config['seed'], shuffled=shuffled)
    readout, diag = fit_stable_rule(
        projected, labels, weights, delta, pw, feature_names=basis_names(FEATURE_NAMES),
        strength=strength, ridge=config['ridge'], scale_floor=config['scale_floor'], manifest_sha=manifest_sha)
    rule = ConditionalRule(gate, readout)
    views = {}
    for variant in ('raw', TRAINING_VARIANT):
        for processed in (False, True):
            views.update({k+'/'+variant: v for k, v in feature_views(
                rows, FEATURE_NAMES, 'threshold', processed=processed, variant=variant).items()})
    rule, calibration = class_threshold(rule, views)
    return rule, {**diag, 'fit_records': len(identities), 'feature_count': len(readout.feature_names),
                  'gate_diagnostics': gate_diag, 'calibration': calibration}
