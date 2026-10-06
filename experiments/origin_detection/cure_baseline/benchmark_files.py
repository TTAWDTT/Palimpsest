"""Time the unchanged CuRe author file path on the established native queue."""

from collections import defaultdict
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
    rows = sorted((r for r in read_rows(OUTPUT / 'scores.csv') if r['variant'] == 'raw' and r['filename'] in names),
                  key=lambda r: r['filename'])
    if len(rows) != 120 or len(prior['measurements']) != 360:
        raise ValueError('CuRe timing denominator changed')
    for row in rows:
        if file_sha256(image_path(row)) != row['sha256']:
            raise ValueError('CuRe timing pixel digest differs')
    model = detector()
    model.predict_file(image_path(rows[0]))
    rng = random.Random(20261006)
    measurements = []
    for repeat in range(3):
        queue = list(rows)
        rng.shuffle(queue)
        for row in queue:
            result = model.predict_file(image_path(row))
            if result.probability_fake != float(row['probability_fake']):
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
        'scope': 'Warm native single file;decode+author preprocessing+GPU+head;torch threads4;'
                 'hashes pre-read;startup excluded;not threads1 custom comparison or phone runtime'})
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
