import numpy as np
import pytest
from palimpsest.detection.baselines.benford import BenfordRFDetector
from palimpsest.detection.baselines.benford_features import image_features


def test_features_have_declared_shape_and_remain_finite_on_flat_image():
    features = image_features(np.full((128, 256, 3), 128, np.uint8))
    assert features.shape == (18,)
    assert np.isfinite(features).all()


def test_numeric_forest_export_matches_sklearn_and_roundtrips(tmp_path):
    sklearn = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(812)
    features = rng.random((60, 18), dtype=np.float32)
    labels = (features[:, 0] + features[:, 3] > 1).astype(int)
    # Tiny representation test, not training a research detector.
    forest = sklearn.RandomForestClassifier(
        n_estimators=3, max_depth=4, random_state=17
    ).fit(features, labels)
    detector = BenfordRFDetector.from_estimator(forest)
    probes = rng.random((20, 18), dtype=np.float32)
    np.testing.assert_allclose(
        [detector.score_features(f) for f in probes],
        forest.predict_proba(probes)[:, 1] - 0.5,
        rtol=0,
        atol=1e-15,
    )
    path = tmp_path / "forest.npz"
    detector.save(path)
    restored = BenfordRFDetector.load(path)
    assert restored.score_features(probes[0]) == detector.score_features(probes[0])
    with pytest.raises(FileExistsError):
        detector.save(path)


def test_numeric_forest_rejects_cycle_instead_of_hanging():
    with pytest.raises(ValueError, match="topology"):
        BenfordRFDetector(
            roots=[0], left=[0], right=[0], feature=[0], threshold=[0.5], leaf_ai=[0.5]
        )
