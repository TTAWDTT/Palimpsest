"""Whole source-panel CV over a legacy-prefix-safe five-direction bank."""

import argparse
from pathlib import Path

from palimpsest.detection.models.frozen_features.cure.quantile_score_consistency import QuantileScoreFitter, QuantileScoreRule, calibrate_quantile_score
from palimpsest.detection.algorithms.readouts.source_hinge import MarginFitRefused
from palimpsest.evaluation.calibrated_readout_campaign import CampaignData, CampaignMethod, write_json
from palimpsest.evaluation.panel_readout_campaign import run_panel_campaign
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_score_consistency.run_iteration import code_pins as parent_pins
from experiments.origin_detection.calibrated_score_subspace.run_iteration import class_threshold, score_rule, null_intervals
from experiments.origin_detection.frozen_features.cure.quantile_score_consistency.protocol import FULL, NAMES, CONFIG, VARIANTS, PANEL, inputs, training, calibration_views, wrong_sources

OUTPUT = WORK_DIR/'robust_statistics/quantile_score_consistency'
TESTS = ('tests/detection/test_quantile_score_consistency.py', 'tests/detection/test_semantic_score_consistency.py',
    'tests/evaluation/test_source_panel_crossfit.py', 'tests/detection/test_consistent_source_risk.py')


def code_pins():
    pins = parent_pins()
    paths = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md',
        REPO_ROOT/'docs/research/token_shape_readout_protocol.md',
        REPO_ROOT/'src/palimpsest/detection/models/frozen_features/cure/quantile_score_consistency.py']
    paths += [REPO_ROOT/name for name in TESTS]
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)})
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-controls', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    data = CampaignData(NAMES, VARIANTS, CONFIG['seed'], FULL/'features.json', inputs, training,
        calibration_views, wrong_sources, lambda a, b, c: score_rule(a, b, c, CONFIG), null_intervals)
    method = CampaignMethod((0, .1, 1, 10), (10, 1), TESTS, code_pins,
        lambda manifest: QuantileScoreFitter(NAMES, manifest_sha=manifest),
        lambda rule, views: calibrate_quantile_score(rule, views, class_threshold), QuantileScoreRule.load)
    try:
        run_panel_campaign(OUTPUT, parser.parse_args(), data, method, PANEL)
    except (MarginFitRefused, ValueError) as error:
        if OUTPUT.exists() and not (OUTPUT/'failure_receipt.json').exists():
            write_json(OUTPUT/'failure_receipt.json', {'passed': False, 'error': str(error),
                'code_pins': code_pins(), 'scope': 'Refused gate/solve;not classifier performance evidence'})
        raise


if __name__ == '__main__':
    main()
