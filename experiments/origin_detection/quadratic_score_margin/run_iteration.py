"""Reuse calibrated campaign with nine polynomial terms over three scores."""

import argparse
from pathlib import Path

from palimpsest.detection.models.frozen_features.cure.quadratic_score_margin import QuadraticMarginFitter, QuadraticScoreRule, calibrate_quadratic_score
from palimpsest.detection.algorithms.readouts.source_hinge import MarginFitRefused
from palimpsest.evaluation.calibrated_readout_campaign import (
    CampaignData, CampaignMethod, run_calibrated_campaign, write_json)
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.calibrated_score_subspace.run_iteration import (
    PARENT, NAMES, CONFIG, VARIANTS, inputs, training, inner_views, code_pins as input_pins,
    class_threshold, score_rule, wrong_sources, null_intervals)

OUTPUT = WORK_DIR/'robust_statistics/quadratic_score_margin'
TESTS = ('tests/detection/test_quadratic_score_margin.py', 'tests/detection/test_source_hinge.py',
         'tests/detection/test_source_score_subspace.py', 'tests/evaluation/test_calibrated_source_crossfit.py')


def code_pins():
    pins = input_pins()
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (*TESTS, 'src/palimpsest/detection/models/frozen_features/cure/quadratic_score_margin.py',
        'src/palimpsest/detection/algorithms/readouts/source_hinge.py', 'src/palimpsest/evaluation/calibrated_readout_campaign.py')]
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)})
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-controls', action='store_true'); parser.add_argument('--pilot', action='store_true')
    data = CampaignData(NAMES, VARIANTS, CONFIG['seed'], PARENT/'features.json', inputs, training,
        inner_views, wrong_sources, lambda a, b, c: score_rule(a, b, c, CONFIG), null_intervals)
    method = CampaignMethod((.0001, .001, .01, .1), (.0001, .1), TESTS, code_pins,
        lambda manifest: QuadraticMarginFitter(NAMES, manifest_sha=manifest),
        lambda rule, views: calibrate_quadratic_score(rule, views, class_threshold), QuadraticScoreRule.load)
    try:
        run_calibrated_campaign(OUTPUT, parser.parse_args(), data, method)
    except MarginFitRefused as error:
        write_json(OUTPUT/'failure_receipt.json', {'passed': False, 'diagnostic': error.diagnostic,
            'code_pins': code_pins(), 'scope': 'Refused finite LP;not evidence against detector'})
        raise


if __name__ == '__main__': main()
