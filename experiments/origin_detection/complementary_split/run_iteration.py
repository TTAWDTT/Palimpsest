"""Fixed mean of both complementary source-split learners."""

import argparse
from pathlib import Path

from palimpsest.detection.algorithms.complementary_split import (
    ComplementarySplitFitter, ComplementarySplitRule, calibrate_complementary)
from palimpsest.detection.algorithms.source_hinge import MarginFitRefused
from palimpsest.evaluation.calibrated_readout_campaign import CampaignData, write_json
from palimpsest.evaluation.panel_readout_campaign import run_panel_campaign
from palimpsest.evaluation.training_fit import BudgetedRecordAwareCampaignMethod
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.split_rate_score.run_iteration import code_pins as parent_pins
from experiments.origin_detection.quantile_score_consistency.run_iteration import (
    FULL, NAMES, CONFIG, VARIANTS, PANEL, inputs, training, calibration_views,
    wrong_sources, score_rule, null_intervals)

OUTPUT = WORK_DIR/'robust_statistics/complementary_split'
TESTS = ('tests/detection/test_complementary_split.py', 'tests/detection/test_source_partition.py',
    'tests/evaluation/test_split_readout_audit.py', 'tests/evaluation/test_training_fit.py',
    'tests/detection/test_paired_threshold.py', 'tests/detection/test_rate_penalty.py')


def code_pins():
    pins = parent_pins()
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md',
        REPO_ROOT/'src/palimpsest/detection/algorithms/complementary_split.py',
        REPO_ROOT/'src/palimpsest/evaluation/calibrated_readout_campaign.py']
    files += [REPO_ROOT/name for name in TESTS]
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)})
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-controls', action='store_true'); parser.add_argument('--pilot', action='store_true')
    data = CampaignData(NAMES, VARIANTS, CONFIG['seed'], FULL/'features.json', inputs, training,
        calibration_views, wrong_sources, lambda a, b, c: score_rule(a, b, c, CONFIG), null_intervals)
    method = BudgetedRecordAwareCampaignMethod((0, 10, 100, 1000), (100, 10), TESTS, code_pins,
        lambda manifest: ComplementarySplitFitter(NAMES, variants=PANEL.variants, manifest_sha=manifest),
        lambda rule, views: calibrate_complementary(rule, views, variants=PANEL.variants),
        ComplementarySplitRule.load, expected_banks=24, pilot_banks=2)
    try:
        run_panel_campaign(OUTPUT, parser.parse_args(), data, method, PANEL)
    except (MarginFitRefused, ValueError) as error:
        if OUTPUT.exists() and not (OUTPUT/'failure_receipt.json').exists():
            write_json(OUTPUT/'failure_receipt.json', {'passed': False, 'error': str(error), 'code_pins': code_pins(),
                'scope': 'Numerical/software gate refused;not classifier performance'})
        raise


if __name__ == '__main__':
    main()
