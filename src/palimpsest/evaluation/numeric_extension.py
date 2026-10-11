"""Stream an extended numeric descriptor against a signed parent prefix.

Explicit callbacks own image locations, JPEG materialization and the frozen
extractor. No fitting or method selection occurs in this storage helper.
"""

import csv
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.io.hashing import file_sha256


def check_prefix(feature,reference,probability):
    if (not np.array_equal(feature.values[:len(reference)],reference)
            or feature.probability_fake!=probability):
        raise ValueError('Extended descriptor changed signed parent prefix/probability')


def extract_extension(inventory,parent_rows,parent_values,output,names,extractor,*,
                      resolve_path,variants,resize,jpeg_parameters,repeat_raw=False,validate):
    """Write arrays only after exact prefix and coverage checks; preserve partials."""
    index = {(r['filename'],r['variant']):i for i,r in enumerate(parent_rows)}
    if len(index)!=len(parent_rows):raise ValueError('Duplicate numeric parent row')
    selected = {(r['filename'],v) for r in inventory
                for v in (variants if r['role']=='selection' else variants[:2])}
    if not selected<=set(index):raise ValueError('Missing numeric parent view')
    output.mkdir(parents=True,exist_ok=True)
    if any((output/n).exists() for n in ('vectors.npy','vectors.partial.npy','metadata.csv','query.jpg')):
        raise FileExistsError('Preserve extension cache/partial')
    partial = output/'vectors.partial.npy'
    matrix = np.lib.format.open_memmap(partial,mode='w+',dtype='<f8',shape=(len(selected),len(names)))
    matrix[:]=np.nan;metadata=[];durations=[];cursor=0;repeated=0
    start = perf_counter();scratch = output/'query.jpg'
    for count,row in enumerate(inventory,1):
        path = resolve_path(row)
        if file_sha256(path)!=row['sha256']:raise ValueError('Extension native bytes changed')
        with Image.open(path) as image:
            if image.size!=(int(row['width']),int(row['height'])):raise ValueError('Extension native size changed')
            pixels = np.asarray(image.convert('RGB'))
        for variant in variants if row['role']=='selection' else variants[:2]:
            query = path
            if variant!='raw':
                quality,sampling = jpeg_parameters[variant]
                Image.fromarray(resize(pixels)).save(scratch,format='JPEG',quality=quality,subsampling=sampling)
                query = scratch
            feature = extractor.extract_file(query);i = index[row['filename'],variant]
            check_prefix(feature,parent_values[i],float(parent_rows[i]['probability_fake']))
            if repeat_raw and variant=='raw':
                repeated_feature = extractor.extract_file(query)
                if (not np.array_equal(feature.values,repeated_feature.values)
                        or feature.probability_fake!=repeated_feature.probability_fake):
                    raise ValueError('Extended descriptor repeat differs')
                repeated += 1
            if feature.values.shape!=(len(names),) or not np.isfinite(feature.values).all():
                raise ValueError('Extended descriptor dimension/value differs')
            matrix[cursor]=feature.values;cursor+=1;durations.append(feature.elapsed_ms)
            metadata.append({**row,'variant':variant,'probability_fake':feature.probability_fake,
                'elapsed_ms':feature.elapsed_ms,'statistics_ms':feature.statistics_ms,**feature.diagnostics})
        if count%60==0 or count==len(inventory):
            print({'images':count,'records':cursor,'elapsed_s':perf_counter()-start},flush=True)
    if cursor!=len(selected):raise ValueError('Extension row count differs')
    validate(metadata,matrix,inventory);matrix.flush();del matrix
    if scratch.exists():scratch.unlink()
    partial.rename(output/'vectors.npy')
    with (output/'metadata.csv').open('x',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(metadata[0]));writer.writeheader();writer.writerows(metadata)
    return {'images':len(inventory),'records':cursor,'all_parent_prefix_exact':True,
            'repeated_raw_records':repeated,'cost_median_ms':float(np.median(durations)),
            'cost_tail60_median_ms':float(np.median(durations[-60:])),
            'elapsed_s':perf_counter()-start,'vectors_sha256':file_sha256(output/'vectors.npy'),
            'metadata_sha256':file_sha256(output/'metadata.csv')}
