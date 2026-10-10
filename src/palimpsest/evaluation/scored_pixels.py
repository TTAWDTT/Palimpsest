"""Decode signed files and score explicit encoding views without label inputs."""

from io import BytesIO
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.io.hashing import file_sha256


def score_inventory(inventory, detector, *, resolve_path, resize, variants, jpeg_parameters,
                    progress=None):
    if not inventory or len(set(variants)) != len(variants) or 'raw' not in variants:
        raise ValueError('Invalid score inventory/variants')
    if set(jpeg_parameters) != set(variants) - {'raw'}:
        raise ValueError('Encoding configuration differs')
    if len({r['filename'] for r in inventory}) != len(inventory):
        raise ValueError('Duplicate score inventory filename')
    start = perf_counter()
    rows = []
    for index, record in enumerate(inventory, 1):
        path = resolve_path(record)
        if file_sha256(path) != record['sha256']:
            raise ValueError('Score image digest changed')
        with Image.open(path) as image:
            if image.size != (int(record['width']), int(record['height'])):
                raise ValueError('Score image dimensions changed')
            rgb = np.asarray(image.convert('RGB'))
        for variant in variants:
            if variant == 'raw':
                pixels = rgb
            else:
                quality, subsampling = jpeg_parameters[variant]
                stream = BytesIO()
                Image.fromarray(resize(rgb)).save(stream, format='JPEG', quality=quality, subsampling=subsampling)
                stream.seek(0)
                with Image.open(stream) as encoded:
                    pixels = np.asarray(encoded.convert('RGB'))
            prediction = detector.predict(pixels)
            rows.append({**record, 'variant': variant, 'score': prediction.margin,
                         'predict_ms': prediction.timing_ms['predict_ms']})
        if progress and (index % 30 == 0 or index == len(inventory)):
            progress({'images': index, 'total': len(inventory), 'records': len(rows),
                      'elapsed_s': perf_counter()-start})
    return rows, {'images': len(inventory), 'records': len(rows), 'elapsed_s': perf_counter()-start,
                  'scope': 'Selection scoring wall cost;not decode-inclusive native latency benchmark'}
