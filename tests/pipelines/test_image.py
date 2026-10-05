import numpy as np
from PIL import Image
from palimpsest.contracts import Prediction
from palimpsest.localization.baselines.quadrilateral import QuadrilateralLocator
from palimpsest.pipelines.image import ImageOriginPipeline


class RecordingDetector:
    """Test double records exact received pixels; never used by production CLI."""

    name = "test_double"

    def __init__(self):
        self.inputs = []

    def predict(self, image):
        self.inputs.append(image.copy())
        return Prediction(self.name, 1.0)


def test_file_chain_keeps_source_coordinates_pixels_and_decode_timing(tmp_path):
    image = np.zeros((140, 210, 3), np.uint8)
    image[20:120, 40:180] = 255
    path = tmp_path / "source.png"
    Image.fromarray(image).save(path)
    detector = RecordingDetector()
    result = ImageOriginPipeline(detector, QuadrilateralLocator()).predict_file(path)
    assert len(result.regions) == 1
    x0, y0, x1, y1 = result.regions[0].region.box.xyxy
    np.testing.assert_array_equal(detector.inputs[0], image[y0:y1, x0:x1])
    times = result.timing_ms
    assert times["decode"] > 0
    assert times["end_to_end"] >= sum(
        times[k] for k in ("decode", "localization", "crop", "detection")
    )
    assert result.to_dict()["regions"][0]["ai_probability"] is None


def test_whole_image_is_explicit_and_empty_localization_runs_no_detector():
    detector = RecordingDetector()
    image = np.zeros((80, 120, 3), np.uint8)
    assert ImageOriginPipeline(detector).predict(image).mode == "whole_image"
    count = len(detector.inputs)
    result = ImageOriginPipeline(detector, QuadrilateralLocator()).predict(image)
    assert result.regions == ()
    assert len(detector.inputs) == count
