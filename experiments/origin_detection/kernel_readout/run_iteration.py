"""Fixed Fourier map capacity diagnostic, keeping all source roles frozen."""

import argparse
from fractions import Fraction
import json
from time import perf_counter
import tomllib

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.group_margin import fit_group_margin
from palimpsest.detection.algorithms.kernel_readout import fit_fourier_map, KernelRule
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.conditional_residual.run_iteration import final_target_screen
from experiments.origin_detection.gaussian_readout.run_iteration import signed_inputs, NAMES
from experiments.origin_detection.paired_stability.fit_rules import paired_deltas
from experiments.origin_detection.paired_stability.run_iteration import INTERVENTION
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.residual_statistics.run_iteration import screen_rule, aggregate
from experiments.origin_detection.residual_training.fit_rules import training_arrays, TRAINING_VARIANT
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.threshold_diagnostics.audit_rate_boundaries import audit

CONFIG = REPO_ROOT/'configs/evaluation/kernel_readout.toml'
OUTPUT = WORK_DIR/'robust_statistics/kernel_readout'
INPUT_NAMES = NAMES['hybrid']


def code_pins():
    files = [CONFIG, REPO_ROOT/'src/palimpsest/detection/algorithms/kernel_readout.py',
             REPO_ROOT/'src/palimpsest/detection/algorithms/group_margin.py',
             REPO_ROOT/'tests/detection/test_kernel_readout.py',
             REPO_ROOT/'experiments/origin_detection/gaussian_readout/run_iteration.py']
    directory = REPO_ROOT/'experiments/origin_detection/kernel_readout'
    files.extend(directory.glob('*.py')); files.append(directory/'README.md')
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def ordered_fit(rows, names, config, *, shuffled=False):
    _, y, weights, expected = training_arrays(rows, 'equal_domain_augmented', config['seed'], shuffled=shuffled)
    matrices, records, identities = [], [], []
    for variant in ('raw', TRAINING_VARIANT):
        views = {**feature_views(rows, names, 'fit', processed=False, variant=variant),
                 **feature_views(rows, names, 'fit', processed=True, variant=variant)}
        for selected, values, _ in views.values():
            matrices.append(values); records.extend(selected)
            identities.extend((r['src'], r['domain'], variant) for r in selected)
    if identities != expected:
        raise ValueError('Kernel fit ordering differs')
    return np.concatenate(matrices), y, weights, records


def map_rows(rows, mapper):
    x = np.array([[float(r[n]) for n in INPUT_NAMES] for r in rows])
    values = mapper.transform(x)
    return [{**row, **dict(zip(mapper.output_names, v.tolist()))} for row, v in zip(rows, values)]


def fit_rule(rows, names, objective, config, manifest_sha, *, shuffled=False, calibrate=True):
    x, y, sw, records = ordered_fit(rows, names, config, shuffled=shuffled)
    keys = [tuple(r[k] for k in ('domain', 'scene', 'condition', 'variant'))+(int(label),) for r, label in zip(records, y)]
    unique = sorted(set(keys)); indices = {key: index for index, key in enumerate(unique)}
    if len(unique) != 48:
        raise ValueError('Kernel class groups differ')
    groups = np.array([indices[k] for k in keys])
    delta, pw = paired_deltas(rows, 'fit', ('raw', TRAINING_VARIANT), names)
    rule, diagnostic = fit_group_margin(x, y, sw, groups, delta, pw, feature_names=names,
                                        ridge=config['ridge'], strength=config['pair_strength'] if objective == 'both' else 0,
                                        temperature=config['group_temperature'] if objective == 'both' else 0,
                                        scale_floor=config['scale_floor'], maximum_iterations=config['maximum_iterations'],
                                        gradient_tolerance=config['gradient_tolerance'], manifest_sha=manifest_sha)
    diagnostic.update({'sources': len({r['src'] for r in records}), 'group_keys': unique, 'dimensions': len(names)})
    if calibrate:
        views = {key+'/'+variant: view for variant in ('raw', TRAINING_VARIANT) for processed in (False, True)
                 for key, view in feature_views(rows, names, 'threshold', processed=processed, variant=variant).items()}
        rule, diagnostic['calibration'] = class_threshold(rule, views)
    return rule, diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    path = OUTPUT/('pilot.json' if args.pilot else 'iteration.json')
    if path.exists():
        raise FileExistsError('Preserve Fourier evaluation')
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    controls = json.loads((OUTPUT/'controls.json').read_text(encoding='utf-8'))
    if controls['returncode'] != 0 or controls['test_sha256'] != file_sha256(REPO_ROOT/'tests/detection/test_kernel_readout.py'):
        raise ValueError('Fourier controls failed or changed')
    ordinary, held, receipt = signed_inputs()
    start = perf_counter()
    with threadpool_limits(limits=1):
        x, _, weights, _ = ordered_fit(ordinary, INPUT_NAMES, config)
        mapper = fit_fourier_map(x, weights, feature_names=INPUT_NAMES, frequency_count=config['frequency_count'],
                                 seed=config['seed'], scale_floor=config['scale_floor'])
        rows, unseen = map_rows(ordinary, mapper), map_rows(held, mapper)
        mapping_s = perf_counter()-start
        dimensions = {'fourier': mapper.output_names, 'hybrid_fourier': INPUT_NAMES+mapper.output_names}
        if args.pilot:
            _, diagnostic = fit_rule(rows, dimensions['fourier'], 'erm', config, receipt['inventory_sha256'], calibrate=False)
            write_json(path, {'elapsed_s': perf_counter()-start, 'mapping_s': mapping_s, 'fit_diagnostics': diagnostic,
                              'code_pins': code_pins(), 'scope': 'Fixed map plus Fourier ERM cost pilot; no selection screening'})
            print(json.dumps({'pilot_s': perf_counter()-start}), flush=True)
            return
        results, rules = {}, {}
        candidates = [(rep, objective, False) for rep in config['candidate_representations'] for objective in config['candidate_objectives']]
        candidates.append(('hybrid_fourier', 'both', True))
        for representation, objective, shuffled in candidates:
            mode = representation+'/'+objective+('_source_shuffled_null' if shuffled else '')
            names = dimensions[representation]
            head, diagnostic = fit_rule(rows, names, objective, config, receipt['inventory_sha256'], shuffled=shuffled)
            kernel = KernelRule(mapper, head, representation == 'hybrid_fourier')
            result = screen_rule(rows, names, head, config)
            encoded = aggregate(unseen, names, head, INTERVENTION)
            changes = [(v['balanced_accuracy_at_zero'], encoded[d]['conditions'][c]['balanced_accuracy_at_zero'])
                       for d, z in result['selection_aggregate'].items() for c, v in z['conditions'].items() if c != 'original']
            result['criteria'].update({
                'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
                'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
                'both_encoded_class_accuracies': all(v[k] >= .55 for data in (result['recompression_aggregate'], encoded)
                                                     for z in data.values() for v in z['conditions'].values()
                                                     for k in ('real_accuracy_at_zero', 'fake_accuracy_at_zero'))})
            result.update({'held_encoding_aggregate': encoded, 'fit_diagnostics': diagnostic,
                           'final_target': final_target_screen(result, encoded, config)})
            exact = audit({'candidates': {mode: result}}, Fraction(str(config['maximum_reencoded_ba_drop'])))[mode]
            result['floating_point_criteria'] = result['criteria']; result['criteria'] = exact['corrected_criteria']
            result['gate_passed'] = exact['corrected_gate_passed']; result['rate_boundary_audit'] = exact['comparisons']
            for variant, data in [('raw', rows), (TRAINING_VARIANT, rows), (INTERVENTION, unseen)]:
                selected = [r for r in data if r['role'] == 'selection' and r['variant'] == variant]
                native = np.array([[float(r[n]) for n in INPUT_NAMES] for r in selected])
                mapped = np.array([[float(r[n]) for n in names] for r in selected])
                expected = head.score(mapped)
                if not np.array_equal(kernel.score(native), expected):
                    raise ValueError('Complete kernel score differs from mapped-head score')
                if not np.array_equal(kernel.score(native), [kernel.score([row])[0] for row in native]):
                    raise ValueError('Kernel single/batch score differs')
            rules[mode] = kernel; results[mode] = result
            print(json.dumps({'candidate': mode, 'gate': result['gate_passed'], 'target': result['final_target']}), flush=True)
    null = 'hybrid_fourier/both_source_shuffled_null'
    refused = results[null]['criteria']['eight_auc_lower_bounds']
    passing = [k for k, v in results.items() if k != null and v['gate_passed']]
    chosen = max(passing, key=lambda k: results[k]['worst_processed_auc']) if passing and not refused else None
    for key, rule in rules.items():
        rule.save(OUTPUT/(key.replace('/', '_')+'_rule.json'))
    write_json(path, {'candidates': results, 'development_chosen': chosen, 'interpretation_refused': refused,
                      'elapsed_s': perf_counter()-start, 'mapping_s': mapping_s,
                      'complete_mapped_and_single_batch_exact': True, 'code_pins': code_pins(),
                      'inventory_sha256': receipt['inventory_sha256'], 'controls_sha256': file_sha256(OUTPUT/'controls.json'),
                      'envelope_receipt_sha256': file_sha256(WORK_DIR/'robust_statistics/wavelet_envelopes/features.json'),
                      'rule_files': {k: file_sha256(OUTPUT/(k.replace('/', '_')+'_rule.json')) for k in rules},
                      'scope': 'Fixed Fourier capacity diagnostic on exposed development; not independent real validation'})


if __name__ == '__main__':
    main()
