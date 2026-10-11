"""Independent weight totals, paired labels and planted classifier controls."""

from io import BytesIO
from hashlib import sha256
import tomllib

import numpy as np
from PIL import Image
import pytest

from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES, extract_features, resize256
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from experiments.origin_detection.residual_statistics.fit_rules import fit_rule as prior_fit
from experiments.origin_detection.residual_statistics.run_iteration import screen_rule
from experiments.origin_detection.residual_training import run_iteration as runner
from experiments.origin_detection.residual_training.fit_rules import MODES, fit_rule, training_arrays


def plant():
    rows = []
    for domain, scenes, conditions in [('rr', ['all'], ['original', 'transfer', 'redigital']),
                                      ('chimera', ['cat', 'church', 'horse'], ['original', 'mac_iphone', 'lg_blackfly'])]:
        for scene in scenes:
            for role in ('fit', 'threshold', 'selection'):
                for i in range(32):
                    a, b = (i // 8) // 2, (i // 8) % 2
                    values = np.zeros(379)
                    for offset in range(0, 28, 7): values[offset+a] = 1
                    for offset in range(28, 379, 39): values[offset+b] = 1
                    src = f'{domain}/{scene}/{role}/{i}'
                    for condition in conditions:
                        for variant in ('raw', 'jpeg90_444_after_resize256'):
                            rows.append({'filename': src+'/'+condition, 'src': src, 'domain': domain, 'scene': scene,
                                         'role': role, 'condition': condition, 'label': 'FAKE' if a != b else 'REAL',
                                         'sha256': 'plant', 'variant': variant, **dict(zip(FEATURE_NAMES, values))})
    return rows


def test_independent_domain_weight_mass_and_paired_shuffle():
    rows = plant()
    for mode in MODES:
        x, labels, weights, identities = training_arrays(rows, mode, 20261006, shuffled=True)
        expected_records = 768 if mode.endswith('_augmented') else 384
        assert len(x) == expected_records
        expected_rr = 9 if mode.startswith('equal_domain') else 3
        assert sum(w for w, (_, d, _) in zip(weights, identities) if d == 'rr') == pytest.approx(expected_rr)
        assert sum(w for w, (_, d, _) in zip(weights, identities) if d == 'chimera') == pytest.approx(9)
        seen = {}
        for (src, _, _), label in zip(identities, labels):
            assert seen.setdefault(src, label) == label
            assert '/fit/' in src  # No threshold/selection contributes.
        assert len(seen) == 128
    with pytest.raises(ValueError): training_arrays(rows, 'tuned_after_selection', 1)


def test_exact_sixth_control_and_known_nonlinear_answer():
    rows = plant()
    config = tomllib.loads(runner.CONFIG.read_text(encoding='utf-8'))
    prior = prior_fit(rows, FEATURE_NAMES, seed=20261006, manifest_sha='plant')
    for mode in MODES:
        rule, _ = fit_rule(rows, mode, seed=20261006, manifest_sha='plant')
        if mode == 'equal_view_raw': assert rule.payload() == prior.payload()
        result = screen_rule(rows, FEATURE_NAMES, rule, config)
        assert result['gate_passed']
        assert all(v['auc'] == 1 and v['balanced_accuracy_at_zero'] == 1 for v in result['selection_views'].values())
        # Missing the residual bit destroys XOR, even with the same trained rule.
        broken = [{**r, **dict.fromkeys(FEATURE_NAMES[28:], 0)} for r in rows]
        assert not screen_rule(broken, FEATURE_NAMES, rule, config)['gate_passed']


def test_eval_only_jpeg_bytes_and_audit_rejections(tmp_path, monkeypatch):
    y, x = np.mgrid[:37, :49]
    rgb = np.stack([(3*x+y)%256, (9*y)%256, (x*17)%256], axis=-1).astype(np.uint8)
    path = tmp_path/'source.png'; Image.fromarray(rgb).save(path)
    row = {'filename': 'plant.png', 'src': 'plant', 'domain': 'rr', 'scene': 'all', 'condition': 'original',
           'role': 'selection', 'label': 'REAL', 'sha256': file_sha256(path), 'width': 49, 'height': 37}
    monkeypatch.setattr(runner, 'image_path', lambda _: path)
    parent = tmp_path/'parent'; parent.mkdir()
    (parent/'features.json').write_text('{"independent_test_receipt": true}', encoding='utf-8')
    monkeypatch.setattr(runner, 'PARENT', parent)
    result = runner.extract_intervention([row], tmp_path)
    records = read_rows(tmp_path/'features.csv')
    assert result['images'] == 1
    stream = BytesIO(); Image.fromarray(resize256(rgb)).save(stream, format='JPEG', quality=70, subsampling=2)
    assert records[0]['jpeg_sha256'] == sha256(stream.getvalue()).hexdigest()
    stream.seek(0)
    with Image.open(stream) as image: expected = extract_features(np.asarray(image.convert('RGB'))).values
    np.testing.assert_array_equal([float(records[0][n]) for n in FEATURE_NAMES], expected)
    for key, value in [('role', 'fit'), ('label', 'FAKE'), ('variant', 'raw'), ('jpeg_sha256', 'bad'),
                       (FEATURE_NAMES[0], 'nan'), (FEATURE_NAMES[0], '0.314')]:
        with pytest.raises(ValueError): runner.audit_intervention([{**records[0], key: value}], [row])
    with pytest.raises(ValueError): runner.audit_intervention(records*2, [row])
    with pytest.raises(ValueError): runner.audit_intervention([], [row])
    for bad in [{**row, 'role': 'fit'}, {**row, 'sha256': 'bad'}, {**row, 'width': 48}]:
        with pytest.raises(ValueError): runner.extract_intervention([bad], tmp_path)
