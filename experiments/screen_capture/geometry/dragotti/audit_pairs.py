"""Audit a small same-content, cross-camera Dragotti recapture probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def read_rgb(path: Path):
    return np.asarray(Image.open(path).convert("RGB"))


def scaled_gray(rgb, max_dim=1200):
    h, w = rgb.shape[:2]
    scale = min(1, max_dim / max(h, w))
    small = cv2.resize(
        rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
    )
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), scale


def align(source, target):
    src, src_scale = scaled_gray(source)
    dst, dst_scale = scaled_gray(target)
    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(src, None)
    k2, d2 = sift.detectAndCompute(dst, None)
    if d1 is None or d2 is None:
        raise RuntimeError("no SIFT descriptors")
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in matches if a.distance < 0.75 * b.distance]
    if len(good) < 20:
        raise RuntimeError(f"only {len(good)} ratio matches")
    p1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h_small, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if h_small is None or mask is None or int(mask.sum()) < 15:
        raise RuntimeError("homography fit failed")
    h_full = (
        np.diag([1 / dst_scale, 1 / dst_scale, 1])
        @ h_small
        @ np.diag([src_scale, src_scale, 1])
    )
    pred = cv2.perspectiveTransform(p1, h_small)
    err = np.linalg.norm(pred - p2, axis=2).flatten()
    inlier = mask.flatten().astype(bool)
    return h_full, {
        "sift_ratio_matches": len(good),
        "ransac_inliers": int(inlier.sum()),
        "inlier_fraction": float(inlier.mean()),
        "median_reprojection_px_at_1200": float(np.median(err[inlier])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--contact-sheet", type=Path, required=True)
    args = p.parse_args()

    source = read_rgb(args.source)
    listing = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    source_thumb = Image.fromarray(source)
    source_thumb.thumbnail((530, 350), Image.Resampling.LANCZOS)
    thumbs = [("Original: Nikon D40", source_thumb)]
    for row in listing["records"]:
        path = Path(row["file"])
        with Image.open(path) as encoded:
            png_info_keys = sorted(encoded.info.keys())
        target = read_rgb(path)
        h, registration = align(source, target)
        center = np.array([source.shape[1] / 2, source.shape[0] / 2, 1.0])
        one_column = center + np.array([1.0, 0.0, 0.0])
        one_row = center + np.array([0.0, 1.0, 0.0])
        center_out, column_out, row_out = h @ center, h @ one_column, h @ one_row
        center_out /= center_out[2]
        column_out /= column_out[2]
        row_out /= row_out[2]
        local_horizontal_scale = float(np.linalg.norm((column_out - center_out)[:2]))
        local_vertical_scale = float(np.linalg.norm((row_out - center_out)[:2]))
        corners = np.float32(
            [
                [[0, 0]],
                [[source.shape[1] - 1, 0]],
                [[source.shape[1] - 1, source.shape[0] - 1]],
                [[0, source.shape[0] - 1]],
            ]
        )
        projected = cv2.perspectiveTransform(corners, h).reshape(4, 2)
        camera = path.name.split("%EA232WMI")[0].split("%")[-1]
        records.append(
            {
                "camera": camera,
                "file": str(path),
                "sha256": row["sha256"],
                "width": target.shape[1],
                "height": target.shape[0],
                "png_info_keys": png_info_keys,
                "registration": registration,
                "center_horizontal_scale_output_per_source_px": local_horizontal_scale,
                "center_vertical_scale_output_per_source_px": local_vertical_scale,
                "horizontal_to_vertical_scale_ratio": local_horizontal_scale
                / local_vertical_scale,
                "conditional_projected_screen_pitch_if_fit_1080_rows": local_vertical_scale
                * source.shape[0]
                / 1080,
                "projected_source_corners": projected.round(2).tolist(),
            }
        )
        thumb = Image.fromarray(target)
        thumb.thumbnail((530, 350), Image.Resampling.LANCZOS)
        thumbs.append((camera, thumb))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "source_file": str(args.source),
                "source_size": [source.shape[1], source.shape[0]],
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if thumbs:
        sheet = Image.new("RGB", (3 * 550, 3 * 390), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (camera, thumb) in enumerate(thumbs):
            x, y = (i % 3) * 550 + 10, (i // 3) * 390 + 30
            draw.text(
                (x, y - 22),
                camera if i == 0 else f"Recapture camera: {camera}",
                fill="black",
            )
            sheet.paste(thumb, (x, y))
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.contact_sheet, optimize=True)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
