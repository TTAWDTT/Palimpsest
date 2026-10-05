import numpy as np
from palimpsest.contracts import Box, Region
from palimpsest.localization.baselines.quadrilateral import QuadrilateralLocator
from palimpsest.evaluation.localization import box_iou, evaluate_regions


def test_contours_find_two_separate_surfaces_and_do_not_fallback():
    image = np.zeros((300, 480, 3), np.uint8)
    image[30:230, 30:180] = 255
    image[70:260, 270:430] = 255
    locator = QuadrilateralLocator()
    regions = locator.locate(image)
    assert len(regions) == 2
    targets = (
        Region("a", Box(30, 30, 180, 230), "surface"),
        Region("b", Box(270, 70, 430, 260), "surface"),
    )
    report = evaluate_regions(regions, targets)
    assert report["recall"] == 1
    assert report["mean_matched_iou"] > 0.96
    for region in regions:
        region.validate_shape(image.shape[:2])
    assert locator.locate(np.zeros_like(image)) == ()


def test_one_to_one_matching_penalizes_duplicate_proposals():
    a = Region("a", Box(0, 0, 10, 10), "surface")
    duplicate = Region("duplicate", a.box, "surface")
    report = evaluate_regions((a, duplicate), (a,))
    assert report["recall"] == 1 and report["precision"] == 0.5
    assert box_iou(a.box, Box(10, 0, 20, 10)) == 0
    assert evaluate_regions((), (a,))["recall"] == 0
