"""Native FP32 versus receptive-field tiling, before real tiled inference."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from palimpsest.detection.baselines.dear import DearRDetector
from palimpsest.detection.baselines.spatial_tiling import tiled_feature_mean
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import MODELS_ROOT, REPO_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.dear_baseline.run_iteration import OUTPUT, VENDOR, source_pins


def main():
    destination = OUTPUT / 'tiling_controls.json'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError('Preserve tiling gate')
    detector = DearRDetector(VENDOR, MODELS_ROOT / 'dear/dear_r.pth', source_pins=source_pins())
    torch = detector.torch
    model = detector.detector.model
    rng = np.random.default_rng(20261006)
    measurements = []
    with threadpool_limits(limits=1), torch.inference_mode():
        for height, width in ((97, 133), (191, 197), (645, 653)):
            image = rng.integers(0, 256, (height, width, 3), np.uint8)
            tensor = detector.transform(Image.fromarray(image)).unsqueeze(0).to(detector.device)
            features = model.forward_features_no_gate(tensor)
            native_mean = features.mean(dim=(2, 3), keepdim=True)
            native_score = model(tensor)
            tiled_mean = tiled_feature_mean(tensor, model.forward_features_no_gate)
            tiled_score = model.backbone.forward_head(model.gate(tiled_mean))
            delta = float((native_score-tiled_score).abs().max().item())
            mean_delta = float((native_mean-tiled_mean).abs().max().item())
            allowed = torch.allclose(native_mean, tiled_mean, atol=1e-3, rtol=1e-5)
            measurements.append({'height': height, 'width': width, 'score_difference': delta,
                                 'mean_maximum_difference': mean_delta, 'mean_tolerance_passed': allowed})
            if delta > 1e-4 or not allowed:
                write_json(destination, {'passed': False, 'measurements': measurements,
                           'scope': 'Predeclared tolerance refused;no tolerance widening'})
                raise ValueError('Tiled/native numerical gate refused')
        bad = tiled_feature_mean(tensor, model.forward_features_no_gate, halo=8, cells=16)
        negative_delta = float((native_mean-bad).abs().max().item())
        if negative_delta <= 1e-3:
            raise ValueError('Undersized halo control did not detect a difference')
    pins = {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in [Path(__file__),
            REPO_ROOT / 'src/palimpsest/detection/baselines/dear.py',
            REPO_ROOT / 'src/palimpsest/detection/baselines/spatial_tiling.py']}
    write_json(destination, {'passed': True, 'measurements': measurements,
        'negative_halo8_mean_difference': negative_delta, 'encoder': detector.provenance,
        'code_pins': pins, 'scope': 'Synthetic numerical execution gate;not real image equivalence or accuracy',
        'registered_score_absolute_tolerance': 1e-4, 'registered_mean_atol': 1e-3,
        'registered_mean_rtol': 1e-5, 'receptive_field': 113, 'stride': 8, 'halo': 64,
        'summation': 'FP64 then FP32;not bitwise native equivalence'})
    print(json.dumps({'passed': True, 'measurements': measurements, 'negative_delta': negative_delta}), flush=True)


if __name__ == '__main__':
    main()
