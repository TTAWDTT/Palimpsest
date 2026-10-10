"""Fixed quadratic readout with registered raw/Q90/Q60 fit-cal panel."""

import argparse
from pathlib import Path

from palimpsest.detection.models.frozen_features.cure.quadratic_score_margin import QuadraticMarginFitter, QuadraticScoreRule, calibrate_quadratic_score
from palimpsest.detection.algorithms.readouts.source_hinge import MarginFitRefused
from palimpsest.evaluation.calibrated_readout_campaign import CampaignData, CampaignMethod, write_json
from palimpsest.evaluation.panel_readout_campaign import run_panel_campaign
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.quadratic_score_margin.run_iteration import code_pins as method_pins
from experiments.origin_detection.calibrated_score_subspace.run_iteration import class_threshold, score_rule, null_intervals
from .protocol import (
    INCREMENT, NAMES, CONFIG, VARIANTS, PANEL, inputs, training, calibration_views, wrong_sources)

OUTPUT = WORK_DIR/'robust_statistics/strong_score_margin'
TESTS = ('tests/evaluation/test_source_panel_crossfit.py', 'tests/detection/test_quadratic_score_margin.py',
         'tests/detection/test_source_hinge.py', 'tests/detection/test_source_score_subspace.py')


def code_pins():
    pins = method_pins()
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (*TESTS, 'src/palimpsest/evaluation/source_panel_crossfit.py',
        'src/palimpsest/evaluation/panel_readout_campaign.py', 'tools/audit_source_panel_cv.py')]
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)})
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-controls', action='store_true'); parser.add_argument('--pilot', action='store_true')
    data = CampaignData(NAMES, VARIANTS, CONFIG['seed'], INCREMENT/'features.json', inputs, training,
        calibration_views, wrong_sources, lambda a, b, c: score_rule(a, b, c, CONFIG), null_intervals)
    method = CampaignMethod((.0001, .001, .01, .1), (.0001, .1), TESTS, code_pins,
        lambda manifest: QuadraticMarginFitter(NAMES, manifest_sha=manifest),
        lambda rule, views: calibrate_quadratic_score(rule, views, class_threshold), QuadraticScoreRule.load)
    try: run_panel_campaign(OUTPUT, parser.parse_args(), data, method, PANEL)
    except MarginFitRefused as error:
        write_json(OUTPUT/'failure_receipt.json', {'passed': False, 'diagnostic': error.diagnostic,
            'code_pins': code_pins(), 'scope': 'Refused solver;not detector evidence'})
        raise


if __name__ == '__main__': main()
