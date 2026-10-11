"""Signed image extraction for new descriptors, with explicit JPEG variants.

The caller chooses inventory, names, variants and extractor. This module does
not select data or fit a classifier. Native RGB decode uses stored orientation.
"""

import csv
from io import BytesIO
import json
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.io.hashing import file_sha256
from .features import IDENTITY_FIELDS, validate_feature_cache


def audit_variants(rows, inventory, names, *, ordinary_variants, selection_variant, bounds=(0, 1)):
    ordinary = [r for r in rows if r['variant'] != selection_variant]
    selected = [r for r in rows if r['variant'] == selection_variant]
    validate_feature_cache(ordinary, inventory, names, variants=ordinary_variants, bounds=bounds)
    subset = [r for r in inventory if r['role'] == 'selection']
    if subset:
        validate_feature_cache(selected, subset, names, variants=(selection_variant,), bounds=bounds)
    elif selected:
        raise ValueError('Unexpected selection-only features')
    return ordinary, selected


def join_features(previous, new, names):
    lookup = {(r['filename'], r['variant']): r for r in new}
    if len(lookup) != len(new) or set(lookup) != {(r['filename'], r['variant']) for r in previous}:
        raise ValueError('New feature coverage differs')
    result = []
    for r in previous:
        extra = lookup[r['filename'], r['variant']]
        if any(r[k] != extra[k] for k in IDENTITY_FIELDS):
            raise ValueError('New feature identity differs')
        result.append({**r, **{n: extra[n] for n in names}})
    return result


def extract_inventory(inventory, output, *, names, extractor, resolve_path, resize,
                      ordinary_variants, selection_variant, jpeg_parameters):
    if output.exists():
        raise FileExistsError('Preserve pixel feature cache')
    start = perf_counter(); rows = []
    for index, r in enumerate(inventory, 1):
        path = resolve_path(r)
        if file_sha256(path) != r['sha256']:
            raise ValueError('Pixel source SHA changed')
        with Image.open(path) as image:
            if image.size != (int(r['width']), int(r['height'])):
                raise ValueError('Pixel source dimensions changed')
            rgb = np.asarray(image.convert('RGB'))
        variants = ordinary_variants + ((selection_variant,) if r['role'] == 'selection' else ())
        for variant in variants:
            if variant == 'raw':
                pixels = rgb
            else:
                quality, subsampling = jpeg_parameters[variant]
                stream = BytesIO()
                Image.fromarray(resize(rgb)).save(stream, format='JPEG', quality=quality, subsampling=subsampling)
                stream.seek(0)
                with Image.open(stream) as image:
                    pixels = np.asarray(image.convert('RGB'))
            value = extractor(pixels)
            rows.append({**r, 'variant': variant, **dict(zip(names, value.values.tolist())),
                         'preprocess_ms': value.preprocess_ms, 'statistics_ms': value.statistics_ms})
        if index % 60 == 0 or index == len(inventory):
            print(json.dumps({'images': index, 'total': len(inventory), 'records': len(rows),
                              'elapsed_s': round(perf_counter()-start, 3)}), flush=True)
    audit_variants(rows, inventory, names, ordinary_variants=ordinary_variants, selection_variant=selection_variant)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return {'images': len(inventory), 'records': len(rows), 'elapsed_s': perf_counter()-start,
            'csv_sha256': file_sha256(output)}
