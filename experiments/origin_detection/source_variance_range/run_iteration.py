"""Extend source variance strengths from the prior OOF boundary point."""

import argparse
from pathlib import Path

from palimpsest.detection.models.frozen_features.cure.semantic_score_consistency import SemanticConsistencyFitter
from palimpsest.detection.models.frozen_features.cure.semantic_score_margin import SemanticScoreRule, calibrate_semantic_score
from palimpsest.detection.algorithms.readouts.source_hinge import MarginFitRefused
from palimpsest.evaluation.calibrated_readout_campaign import CampaignData, CampaignMethod, write_json
from palimpsest.evaluation.panel_readout_campaign import run_panel_campaign
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_score_consistency.run_iteration import code_pins as parent_pins, TESTS
from experiments.origin_detection.calibrated_score_subspace.run_iteration import class_threshold, score_rule, null_intervals
from experiments.origin_detection.semantic_score_margin.protocol import (
    INCREMENT, NAMES, CONFIG, VARIANTS, PANEL, inputs, training, calibration_views, wrong_sources)

OUTPUT = WORK_DIR/'robust_statistics/source_variance_range'
PARAMETERS = (10, 30, 100, 300)


def code_pins():
    pins = parent_pins(); paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)})
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-controls', action='store_true'); parser.add_argument('--pilot', action='store_true')
    data = CampaignData(NAMES, VARIANTS, CONFIG['seed'], INCREMENT/'features.json', inputs, training,
        calibration_views, wrong_sources, lambda a, b, c: score_rule(a, b, c, CONFIG), null_intervals)
    method = CampaignMethod(PARAMETERS, (300, 30), TESTS, code_pins,
        lambda manifest: SemanticConsistencyFitter(NAMES, manifest_sha=manifest),
        lambda rule, views: calibrate_semantic_score(rule, views, class_threshold), SemanticScoreRule.load)
    try: run_panel_campaign(OUTPUT, parser.parse_args(), data, method, PANEL)
    except (MarginFitRefused, ValueError) as error:
        if OUTPUT.exists() and not (OUTPUT/'failure_receipt.json').exists():
            write_json(OUTPUT/'failure_receipt.json', {'passed': False, 'error': str(error),
                'code_pins': code_pins(), 'scope': 'Refused gate/solve;not detector evidence'})
        raise


if __name__ == '__main__': main()
