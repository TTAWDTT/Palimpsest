"""Fast OpenCV contour proposals for visibly bounded planar surfaces.

This is a reference heuristic, not semantic object recognition. It can propose
picture frames, screens and paper as well as unrelated quadrilaterals. Thresholds
are explicit parameters and have not been calibrated on a localization dataset.
"""

import cv2
import numpy as np
from palimpsest.contracts import Box, RGBImage, Region, validate_rgb


class QuadrilateralLocator:
    name = "opencv_quadrilateral"

    def __init__(self, *, min_area_fraction=0.02, max_regions=8, max_side=1280):
        if not 0 < min_area_fraction < 1 or max_regions < 1 or max_side < 32:
            raise ValueError("Invalid quadrilateral proposal parameters")
        self.min_area_fraction = min_area_fraction
        self.max_regions = max_regions
        self.max_side = max_side

    def locate(self, image: RGBImage) -> tuple[Region, ...]:
        validate_rgb(image)
        height, width = image.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        small = (
            cv2.resize(
                image, (max(1, round(width * scale)), max(1, round(height * scale)))
            )
            if scale < 1
            else image
        )
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_fraction * gray.size:
                continue
            polygon = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True
            )
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            # Map back using actual rounded dimensions rather than assumed scale.
            xy = polygon[:, 0, :].astype(np.float64)
            xy *= np.array([width / small.shape[1], height / small.shape[0]])
            xy = np.rint(xy).astype(np.int32)
            xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, xy, 1)
            ys, xs = np.nonzero(mask)
            box = Box(
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            )
            candidates.append((int(mask.sum()), box, mask.astype(bool)))
        candidates.sort(key=lambda item: (-item[0], item[1].xyxy))
        selected = []
        for _, box, mask in candidates:
            # Nested edge contours produce duplicates; do not discard distant boxes.
            duplicate = False
            for region in selected:
                other = region.box
                intersection = max(
                    0, min(box.x1, other.x1) - max(box.x0, other.x0)
                ) * max(0, min(box.y1, other.y1) - max(box.y0, other.y0))
                if intersection / (box.area + other.area - intersection) > 0.9:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(
                    Region(
                        str(len(selected)),
                        box,
                        "quadrilateral_surface_candidate",
                        mask=mask,
                    )
                )
            if len(selected) == self.max_regions:
                break
        return tuple(selected)
