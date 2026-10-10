"""Pinned existing CLIP representation; source roles never changed."""

import argparse
import json
import random
from time import perf_counter
import tomllib

import cv2
from threadpoolctl import threadpool_limits

from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES, FrozenClip
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.pixel_features import audit_variants, extract_inventory, join_features
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.frozen_readout.evaluate_representation import fit_and_screen
from experiments.origin_detection.gaussian_readout.run_iteration import signed_inputs, NAMES
from experiments.origin_detection.paired_stability.run_iteration import INTERVENTION
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.residual_training.fit_rules import TRAINING_VARIANT

CONFIG = REPO_ROOT/'configs/evaluation/frozen_clip.toml'
OUTPUT = WORK_DIR/'robust_statistics/frozen_clip'
ORDINARY = ('raw', TRAINING_VARIANT)
MODES = {'clip': FEATURE_NAMES, 'hybrid_clip': NAMES['hybrid']+FEATURE_NAMES}


def code_pins():
    files = [CONFIG, REPO_ROOT/'src/palimpsest/detection/representations/frozen_clip.py',
             REPO_ROOT/'src/palimpsest/evaluation/pixel_features.py',
             REPO_ROOT/'tests/detection/test_frozen_clip.py',
             REPO_ROOT/'experiments/origin_detection/frozen_readout/evaluate_representation.py',
             REPO_ROOT/'experiments/origin_detection/kernel_readout/run_iteration.py',
             REPO_ROOT/'experiments/origin_detection/gaussian_readout/run_iteration.py']
    directory = REPO_ROOT/'experiments/origin_detection/frozen_clip'
    files.extend(directory.glob('*.py')); files.append(directory/'README.md')
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', type=int, default=0)
    group.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.pilot < 0: parser.error('Pilot must be positive')
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    control_path = OUTPUT/'controls.json'
    control = json.loads(control_path.read_text(encoding='utf-8'))
    if control['returncode'] != 0 or control['test_sha256'] != file_sha256(REPO_ROOT/'tests/detection/test_frozen_clip.py'):
        raise ValueError('CLIP controls failed or changed')
    ordinary, held, parent = signed_inputs()
    removed = set(NAMES['hybrid']) | {'preprocess_ms', 'statistics_ms', 'variant'}
    inventory = [{k: v for k, v in r.items() if k not in removed} for r in ordinary if r['variant'] == 'raw']
    cv2.setNumThreads(1)
    if args.prepare or args.pilot:
        directory = OUTPUT if args.prepare else OUTPUT.with_name('frozen_clip_pilot')
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/'features.json').exists(): raise FileExistsError('Preserve CLIP extraction receipt')
        encoder = FrozenClip(MODELS_ROOT/'d3/ViT-L-14.pt', device=config['device'])
        if args.pilot:
            random.Random(config['seed']).shuffle(inventory); inventory = inventory[:args.pilot]
        result = extract_inventory(inventory, directory/'features.csv', names=FEATURE_NAMES, extractor=encoder.extract,
                                   resolve_path=image_path, resize=resize256, ordinary_variants=ORDINARY,
                                   selection_variant=INTERVENTION, jpeg_parameters={TRAINING_VARIANT: (90, 0), INTERVENTION: (70, 2)})
        write_json(directory/'features.json', {**result, 'code_pins': code_pins(), 'encoder': encoder.provenance,
                    'inventory_sha256': parent['inventory_sha256'], 'controls_sha256': file_sha256(control_path),
                    'parent_features_receipt_sha256': file_sha256(WORK_DIR/'robust_statistics/wavelet_envelopes/features.json')})
        return
    if (OUTPUT/'iteration.json').exists(): raise FileExistsError('Preserve CLIP evaluation')
    receipt = json.loads((OUTPUT/'features.json').read_text(encoding='utf-8'))
    if (receipt['code_pins'] != code_pins() or receipt['csv_sha256'] != file_sha256(OUTPUT/'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256']):
        raise ValueError('CLIP code/cache/inventory changed')
    raw, extra = audit_variants(read_rows(OUTPUT/'features.csv'), inventory, FEATURE_NAMES,
                                ordinary_variants=ORDINARY, selection_variant=INTERVENTION)
    rows, unseen = join_features(ordinary, raw, FEATURE_NAMES), join_features(held, extra, FEATURE_NAMES)
    start = perf_counter(); results = {}; rules = {}
    candidates = [(rep, objective, False) for rep in config['candidate_representations'] for objective in config['candidate_objectives']]
    candidates.append(('hybrid_clip', 'both', True))
    with threadpool_limits(limits=1):
        for rep, objective, shuffled in candidates:
            key = rep+'/'+objective+('_source_shuffled_null' if shuffled else '')
            rule, result = fit_and_screen(rows, unseen, MODES[rep], objective, config, parent['inventory_sha256'], shuffled=shuffled)
            results[key] = result; rules[key] = rule
            print(json.dumps({'candidate': key, 'gate': result['gate_passed'], 'target': result['final_target']}), flush=True)
    null = 'hybrid_clip/both_source_shuffled_null'
    refused = results[null]['criteria']['eight_auc_lower_bounds']
    passing = [k for k, v in results.items() if k != null and v['gate_passed']]
    chosen = max(passing, key=lambda k: results[k]['worst_processed_auc']) if passing and not refused else None
    for key, rule in rules.items(): rule.save(OUTPUT/(key.replace('/', '_')+'_rule.json'))
    write_json(OUTPUT/'iteration.json', {'candidates': results, 'development_chosen': chosen,
                'interpretation_refused': refused, 'elapsed_s': perf_counter()-start,
                'single_batch_scores_exact': True, 'code_pins': code_pins(),
                'inventory_sha256': parent['inventory_sha256'], 'controls_sha256': file_sha256(control_path),
                'features_receipt_sha256': file_sha256(OUTPUT/'features.json'),
                'rule_files': {k: file_sha256(OUTPUT/(k.replace('/', '_')+'_rule.json')) for k in rules},
                'scope': 'Existing frozen neural representation on exposed development, not independent validation'})


if __name__ == '__main__': main()
