"""Same bank and paired calibration; train a bounded error-rate proxy head."""

import argparse
from pathlib import Path

from palimpsest.detection.models.frozen_features.cure.rate_score_readout import RateScoreFitter
from palimpsest.detection.models.frozen_features.cure.quantile_score_consistency import QuantileScoreRule
from palimpsest.detection.algorithms.readouts.source_hinge import MarginFitRefused
from palimpsest.evaluation.calibrated_readout_campaign import CampaignData, write_json
from palimpsest.evaluation.panel_readout_campaign import run_panel_campaign
from palimpsest.evaluation.training_fit import RecordAwareCampaignMethod
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.frozen_features.cure.paired_ba_calibration.run_iteration import calibrate, code_pins as parent_pins
from experiments.origin_detection.frozen_features.cure.quantile_score_consistency.run_iteration import FULL, NAMES, CONFIG, VARIANTS, PANEL, inputs, training, calibration_views, wrong_sources, score_rule, null_intervals

OUTPUT = WORK_DIR/'robust_statistics/rate_score_readout'
TESTS = ('tests/detection/test_rate_penalty.py', 'tests/detection/test_paired_threshold.py',
    'tests/evaluation/test_training_fit.py', 'tests/detection/test_quantile_score_consistency.py')


def code_pins():
    pins = parent_pins()
    paths = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md',
        REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/rate_penalty.py',
        REPO_ROOT/'src/palimpsest/detection/models/frozen_features/cure/rate_score_readout.py']
    paths += [REPO_ROOT/name for name in TESTS]
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)})
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-controls', action='store_true'); parser.add_argument('--pilot', action='store_true')
    data = CampaignData(NAMES, VARIANTS, CONFIG['seed'], FULL/'features.json', inputs, training,
        calibration_views, wrong_sources, lambda a, b, c: score_rule(a, b, c, CONFIG), null_intervals)
    method = RecordAwareCampaignMethod((0, 10, 100, 1000), (100, 10), TESTS, code_pins,
        lambda manifest: RateScoreFitter(NAMES, variants=PANEL.variants, manifest_sha=manifest),
        calibrate, QuantileScoreRule.load)
    try:
        run_panel_campaign(OUTPUT, parser.parse_args(), data, method, PANEL)
    except (MarginFitRefused, ValueError) as error:
        if OUTPUT.exists() and not (OUTPUT/'failure_receipt.json').exists():
            write_json(OUTPUT/'failure_receipt.json', {'passed': False, 'error': str(error), 'code_pins': code_pins(),
                'scope': 'Failed numerical/software gate;not classifier performance'})
        raise


if __name__ == '__main__':
    main()
