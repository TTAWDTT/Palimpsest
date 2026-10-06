"""Aligned receptive-field tiles for finite-support convolutional features.

The caller must establish stride/support and absence of global spatial layers.
This computes a global feature mean, not a mean of tile classification scores.
"""

import math


def feature_tiles(height, width, *, stride=8, halo=64, cells=64):
    if (any(type(v) is not int or v <= 0 for v in (height, width, stride, halo, cells))
            or halo % stride):
        raise ValueError('Invalid aligned feature tile dimensions')
    out_height, out_width = math.ceil(height / stride), math.ceil(width / stride)
    for y0 in range(0, out_height, cells):
        y1 = min(y0 + cells, out_height)
        top, bottom = max(0, y0 * stride - halo), min(height, y1 * stride + halo)
        for x0 in range(0, out_width, cells):
            x1 = min(x0 + cells, out_width)
            left, right = max(0, x0 * stride - halo), min(width, x1 * stride + halo)
            yield ((top, bottom, left, right),
                   (y0-top//stride, y1-top//stride, x0-left//stride, x1-left//stride))


def tiled_feature_mean(image, forward_features, *, stride=8, halo=64, cells=64):
    import torch
    if image.ndim != 4 or image.shape[0] != 1 or torch.is_grad_enabled():
        raise ValueError('Tiled feature mean requires single-image inference mode')
    height, width = image.shape[-2:]
    total, count = None, 0
    for (top, bottom, left, right), (y0, y1, x0, x1) in feature_tiles(
            height, width, stride=stride, halo=halo, cells=cells):
        features = forward_features(image[..., top:bottom, left:right])
        if features.shape[-2:] != (math.ceil((bottom-top)/stride), math.ceil((right-left)/stride)):
            raise ValueError('Feature output geometry differs from registered stride')
        selected = features[..., y0:y1, x0:x1]
        if selected.shape[-2:] != (y1-y0, x1-x0):
            raise ValueError('Invalid retained feature tile')
        part = selected.sum(dim=(2, 3), keepdim=True, dtype=torch.float64)
        total = part if total is None else total + part
        count += (y1-y0) * (x1-x0)
    if count != math.ceil(height/stride) * math.ceil(width/stride):
        raise ValueError('Incomplete global feature lattice')
    return (total/count).to(dtype=image.dtype)
