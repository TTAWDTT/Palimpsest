"""Fixed true-CV parameter, same data/policy, new shape input erased at fit."""

import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.models.frozen_features.cure.quantile_score_consistency import QuantileScoreFitter, QuantileScoreRule, calibrate_quantile_score
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from experiments.origin_detection.frozen_features.cure.quantile_score_consistency.run_iteration import OUTPUT, FULL, code_pins, inputs, training, NAMES, PANEL, CONFIG, class_threshold, score_rule, write_json


def main():
    destination = OUTPUT/'masked_shape_ablation.json'
    if destination.exists():
        raise FileExistsError('Preserve matched shape ablation')
    original = json.loads((OUTPUT/'iteration.json').read_text())
    if (original['code_pins'] != code_pins()
            or original['selection_scores_sha256'] != file_sha256(OUTPUT/'selection_scores.json')):
        raise ValueError('Original classifier changed')
    crossfit_path = OUTPUT/'crossfit.json'
    if file_sha256(crossfit_path) != original['crossfit_sha256']:
        raise ValueError('Original source CV changed')
    crossfit = json.loads(crossfit_path.read_text())
    if original['chosen_strengths']['truth'] != crossfit['chosen']['truth']:
        raise ValueError('Ablation parameter is not original true-CV choice')
    parameter = float(original['chosen_strengths']['truth'][0])
    rows, parent = inputs(); _, x, y, weights, sources = training(rows)
    if (original['features_receipt_sha256'] != file_sha256(FULL/'features.json')
            or original['inventory_sha256'] != parent['inventory_sha256']
            or crossfit['inventory_sha256'] != parent['inventory_sha256']):
        raise ValueError('Ablation does not use the same signed data')
    x = x.copy(); x[:, 3600:] = 0
    fitter = QuantileScoreFitter(NAMES, manifest_sha=parent['inventory_sha256'])
    views = {key+'/'+variant: view for variant in PANEL.variants for p in (False, True)
        for key, view in feature_views(rows, NAMES, 'threshold', processed=p, variant=variant).items()}
    start = perf_counter()
    with threadpool_limits(limits=1):
        rule, diagnostic = fitter.fit(x, y, weights, sources, parameter)
        if any(value != 0 for value in rule.bank.shape.weights):
            raise ValueError('Zero shape input learned nonzero coefficient')
        rule, calibration = calibrate_quantile_score(rule, views, class_threshold)
        path = OUTPUT/'masked_shape_rule.json'; rule.save(path)
        scores, result = score_rule(rows, NAMES, rule, CONFIG)
        matrix = np.array([[float(r[name]) for name in NAMES] for r in rows if r['role'] == 'selection'])
        if not np.array_equal(rule.score(matrix), QuantileScoreRule.load(path).score(matrix)):
            raise ValueError('Ablation portable mismatch')
    write_json(OUTPUT/'masked_shape_scores.json', {'masked_shape': scores})
    write_json(destination, {'candidates': {'masked_shape': result}, 'chosen_true_cv_parameter': parameter,
        'fit_diagnostic': diagnostic, 'calibration': calibration, 'bank_ledger': fitter.audit(),
        'elapsed_s': perf_counter()-start, 'code_pins': code_pins(),
        'original_iteration_sha256': file_sha256(OUTPUT/'iteration.json'),
        'features_receipt_sha256': file_sha256(FULL/'features.json'),
        'crossfit_sha256': file_sha256(crossfit_path),
        'selection_scores_sha256': file_sha256(OUTPUT/'masked_shape_scores.json'),
        'scope': 'Prespecified matched information ablation;no choice from outer score or independence claim'})


if __name__ == '__main__':
    main()
