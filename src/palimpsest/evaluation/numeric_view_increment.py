"""Extract a single missing JPEG view with optional full numeric reference.

Unlike numeric_extension, new views need not have a parent prefix. A pilot can
require a complete old-vector witness; full runs retain query hashes and native
identity. This helper never fits a classifier or chooses its parameters.
"""

import csv
from time import perf_counter

import numpy as np
from PIL import Image

from .numeric_extension import check_prefix
from palimpsest.io.hashing import file_sha256


def extract_numeric_view(inventory, output, names, extractor, *, variant, quality, sampling,
                         resolve_path, resize, validate, reference=None):
    if not inventory or len({r['filename'] for r in inventory}) != len(inventory):
        raise ValueError('Empty/duplicate view inventory')
    output.mkdir(parents=True, exist_ok=True)
    if any((output/n).exists() for n in ('vectors.npy', 'vectors.partial.npy', 'metadata.csv', 'query.jpg')):
        raise FileExistsError('Preserve incremental numeric cache/partial')
    if reference is not None and set(reference) != {(r['filename'], variant) for r in inventory}:
        raise ValueError('Incremental reference identity differs')
    partial = output/'vectors.partial.npy'
    matrix = np.lib.format.open_memmap(partial, mode='w+', dtype='<f8', shape=(len(inventory), len(names)))
    matrix[:] = np.nan; metadata = []; durations = []; scratch = output/'query.jpg'; start = perf_counter()
    try:
        for index, row in enumerate(inventory):
            path = resolve_path(row)
            if file_sha256(path) != row['sha256']: raise ValueError('Native incremental bytes changed')
            with Image.open(path) as image:
                if image.size != (int(row['width']), int(row['height'])):
                    raise ValueError('Native incremental dimensions changed')
                pixels = np.asarray(image.convert('RGB'))
            Image.fromarray(resize(pixels)).save(scratch, format='JPEG', quality=quality, subsampling=sampling)
            feature = extractor.extract_file(scratch)
            if feature.values.shape != (len(names),) or not np.isfinite(feature.values).all():
                raise ValueError('Incremental feature dimensions/values differ')
            if reference is not None:
                values, probability = reference[row['filename'], variant]
                if len(values) != len(names): raise ValueError('Require complete incremental reference')
                check_prefix(feature, values, probability)
            matrix[index] = feature.values; durations.append(feature.elapsed_ms)
            metadata.append({**row, 'variant': variant, 'query_sha256': file_sha256(scratch),
                'probability_fake': feature.probability_fake, 'elapsed_ms': feature.elapsed_ms,
                'statistics_ms': feature.statistics_ms, **feature.diagnostics})
            if (index+1) % 60 == 0 or index+1 == len(inventory):
                print({'images': index+1, 'records': index+1, 'elapsed_s': perf_counter()-start}, flush=True)
        validate(metadata, matrix, inventory)
        matrix.flush()
    finally:
        del matrix
    # On failure preserve partial and query for diagnosis. Success releases both.
    scratch.unlink(); partial.rename(output/'vectors.npy')
    with (output/'metadata.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0])); writer.writeheader(); writer.writerows(metadata)
    return {'records': len(inventory), 'images': len(inventory), 'elapsed_s': perf_counter()-start,
        'cost_median_ms': float(np.median(durations)), 'cost_tail60_median_ms': float(np.median(durations[-60:])),
        'complete_reference_exact': reference is not None, 'variant': variant, 'quality': quality, 'sampling': sampling,
        'vectors_sha256': file_sha256(output/'vectors.npy'), 'metadata_sha256': file_sha256(output/'metadata.csv')}
