"""Time the unchanged CuRe author file path on the established native queue."""

from collections import Counter, defaultdict
import csv
import json
import random

from palimpsest.evaluation.timing import percentile
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.cure_baseline.run_iteration import OUTPUT, code_pins, detector, runtime


def main():
    destination = OUTPUT / 'benchmark.json'
    if destination.exists():
        raise FileExistsError('Preserve CuRe timing evidence')
    receipt = json.loads((OUTPUT / 'iteration.json').read_text())
    if (receipt['code_pins'] != code_pins() or receipt['runtime'] != runtime()
            or receipt['scores_sha256'] != file_sha256(OUTPUT / 'scores.csv')):
        raise ValueError('CuRe runtime or scoring evidence changed')
    reference = WORK_DIR / 'robust_statistics/frozen_clip/benchmark.json'
    prior = json.loads(reference.read_text())
    names = {r['filename'] for r in prior['measurements']}
    probabilities = {r['filename']: float(r['probability_fake']) for r in read_rows(OUTPUT / 'scores.csv')
                     if r['variant'] == 'raw' and r['filename'] in names}
    parent = WORK_DIR / 'robust_statistics/frozen_clip/features.json'
    signed = json.loads(parent.read_text())
    source = parent.with_suffix('.csv')
    if (signed['csv_sha256'] != file_sha256(source)
            or signed['inventory_sha256'] != receipt['inventory_sha256']):
        raise ValueError('CuRe timing parent metadata changed')
    with source.open(newline='', encoding='utf-8-sig') as stream:
        rows = sorted((r for r in csv.DictReader(stream) if r['variant'] == 'raw' and r['filename'] in names),
                      key=lambda r: r['filename'])
    if len(rows) != 120 or len(prior['measurements']) != 360:
        raise ValueError('CuRe timing denominator changed')
    roles = dict(Counter(r['role'] for r in rows))
    if roles != {'fit': 32, 'selection': 74, 'threshold': 14} or len(probabilities) != 74:
        raise ValueError('CuRe timing role inventory changed')
    for row in rows:
        if file_sha256(image_path(row)) != row['sha256']:
            raise ValueError('CuRe timing pixel digest differs')
    model = detector()
    model.predict_file(image_path(rows[0]))
    for row in rows:
        if row['filename'] not in probabilities:
            probabilities[row['filename']] = model.predict_file(image_path(row)).probability_fake
    rng = random.Random(20261006)
    measurements = []
    for repeat in range(3):
        queue = list(rows)
        rng.shuffle(queue)
        for row in queue:
            result = model.predict_file(image_path(row))
            if result.probability_fake != probabilities[row['filename']]:
                raise ValueError('CuRe cached/live probability changed')
            measurements.append({'filename': row['filename'], 'repeat': repeat,
                'pixels': int(row['width']) * int(row['height']),
                'decode_preprocess_ms': result.decode_preprocess_ms,
                'forward_ms': result.forward_ms, 'end_to_end_ms': result.end_to_end_ms})
    groups = defaultdict(list)
    for row in measurements:
        groups['all'].append(row)
        groups['at_most_2MP' if row['pixels'] <= 2_000_000 else 'above_2MP'].append(row)
    summaries = {}
    for key, selected in groups.items():
        summaries[key] = {'images': len({r['filename'] for r in selected}), 'measurements': len(selected),
            **{metric: {f'p{int(q*100)}': percentile([r[metric] for r in selected], q) for q in (.5, .95)}
               for metric in ('decode_preprocess_ms', 'forward_ms', 'end_to_end_ms')}}
    write_json(destination, {'summaries': summaries, 'measurements': measurements,
        'code_pins': code_pins(), 'encoder': model.provenance,
        'iteration_sha256': file_sha256(OUTPUT / 'iteration.json'),
        'reference_benchmark_sha256': file_sha256(reference), 'live_probability_exact': True,
        'reference_probabilities': probabilities, 'timing_only_role_counts': roles,
        'untimed_extra_reference_files': 46,
        'scope': 'Warm native single file;decode+author preprocessing+GPU+head;torch threads4;'
                 'hashes pre-read;startup excluded;not threads1 custom comparison or phone runtime'})
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
