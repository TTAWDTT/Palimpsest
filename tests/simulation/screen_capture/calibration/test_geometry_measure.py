"""Digital preflight for geometry recovery; real camera validation is separate."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from palimpsest.simulation.screen_capture.calibration.capture_kit import generate
from palimpsest.simulation.screen_capture.calibration.geometry_measure import measure


def test_recovers_projection_after_warp_blur_and_jpeg(tmp_path):
    kit = tmp_path / "kit"
    generate(kit, 640, 480)
    source = cv2.imread(str(kit / "frames/geometry.png"))
    src = np.float32([[0, 0], [639, 0], [639, 479], [0, 479]])
    dst = np.float32([[120, 80], [680, 60], [700, 510], [100, 520]])
    true_h = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(source, true_h, (800, 600), borderValue=(30, 30, 30))
    warped = cv2.GaussianBlur(warped, (5, 5), 0.8)
    image = tmp_path / "synthetic_camera.jpg"
    assert cv2.imwrite(str(image), warped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result = measure(kit, image)
    expected_center = cv2.perspectiveTransform(np.float32([[[320, 240]]]), true_h)[0, 0]
    assert result["detected_inner_corners"] == 77
    assert result["homography_inliers"] >= 70
    assert result["marker_color_score"] < 0.08
    assert (
        np.linalg.norm(np.asarray(result["projected_center_xy_px"]) - expected_center)
        < 2
    )
