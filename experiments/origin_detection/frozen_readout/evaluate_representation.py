"""Shared fixed-head screening for prospectively frozen representation trials."""

from fractions import Fraction
import numpy as np

from experiments.origin_detection.conditional_residual.run_iteration import final_target_screen
from experiments.origin_detection.kernel_readout.run_iteration import fit_rule
from experiments.origin_detection.paired_stability.run_iteration import INTERVENTION
from experiments.origin_detection.residual_statistics.run_iteration import screen_rule, aggregate
from experiments.origin_detection.residual_training.fit_rules import TRAINING_VARIANT
from experiments.origin_detection.threshold_diagnostics.audit_rate_boundaries import audit


def fit_and_screen(rows, unseen, names, objective, config, manifest_sha, *, shuffled=False):
    """Fit on fit roles, calibrate on threshold roles, screen selection only."""
    rule, diagnostic = fit_rule(rows, names, objective, config, manifest_sha, shuffled=shuffled)
    result = screen_rule(rows, names, rule, config)
    encoded = aggregate(unseen, names, rule, INTERVENTION)
    changes = [(v['balanced_accuracy_at_zero'], encoded[d]['conditions'][c]['balanced_accuracy_at_zero'])
               for d, z in result['selection_aggregate'].items() for c, v in z['conditions'].items() if c != 'original']
    result['criteria'].update({
        'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
        'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
        'both_encoded_class_accuracies': all(v[k] >= config['minimum_processed_class_accuracy']
                                             for data in (result['recompression_aggregate'], encoded)
                                             for z in data.values() for v in z['conditions'].values()
                                             for k in ('real_accuracy_at_zero', 'fake_accuracy_at_zero'))})
    result.update({'held_encoding_aggregate': encoded, 'fit_diagnostics': diagnostic,
                   'final_target': final_target_screen(result, encoded, config)})
    exact = audit({'candidates': {'candidate': result}}, Fraction(str(config['maximum_reencoded_ba_drop'])))['candidate']
    result['floating_point_criteria'] = result['criteria']; result['criteria'] = exact['corrected_criteria']
    result['gate_passed'] = exact['corrected_gate_passed']; result['rate_boundary_audit'] = exact['comparisons']
    for variant, data in [('raw', rows), (TRAINING_VARIANT, rows), (INTERVENTION, unseen)]:
        matrix = np.array([[float(r[n]) for n in names] for r in data if r['role'] == 'selection' and r['variant'] == variant])
        if not np.array_equal(rule.score(matrix), [rule.score([row])[0] for row in matrix]):
            raise ValueError('Fixed representation single/batch score differs')
    return rule, result
