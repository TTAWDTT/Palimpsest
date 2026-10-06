"""Audit fixed DEAR score coverage and integer rates without another inference."""

import csv
from fractions import Fraction
import json
import math
from pathlib import Path

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.frozen_readout.audit_scored_views import audit
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS


OUTPUT = WORK_DIR / 'robust_statistics/dear_baseline_tiled'
PARENT = WORK_DIR / 'robust_statistics/frozen_clip'


def main():
    destination = OUTPUT / 'score_counts_audit.json'
    if destination.exists():
        raise FileExistsError('Preserve fixed-baseline audit')
    receipt = json.loads((OUTPUT / 'iteration.json').read_text())
    parent = json.loads((PARENT / 'features.json').read_text())
    if (file_sha256(OUTPUT / 'scores.csv') != receipt['scores_sha256']
            or file_sha256(PARENT / 'features.csv') != parent['csv_sha256']
            or parent['inventory_sha256'] != receipt['inventory_sha256']):
        raise ValueError('Scored data or parent receipt changed')
    for relative, expected in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise ValueError('Fixed-baseline execution code changed')
    # Stream the parent metadata; no large feature table or new pixels are loaded.
    with (PARENT / 'features.csv').open(newline='', encoding='utf-8-sig') as stream:
        inventory = {r['filename']: r for r in csv.DictReader(stream)
                     if r['variant'] == 'raw' and r['role'] == 'selection'}
    with (OUTPUT / 'scores.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(name, variant) for name in inventory for variant in VARIANTS}
    actual = [(r['filename'], r['variant']) for r in rows]
    if len(inventory) != 1260 or len(rows) != 5040 or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Fixed-baseline coverage differs')
    metadata = ('src', 'condition', 'label', 'scene', 'domain', 'sha256',
                'width', 'height', 'role', 'relative_path')
    for row in rows:
        if any(row[k] != inventory[row['filename']][k] for k in metadata):
            raise ValueError('Fixed-baseline source metadata differs')
        row['score'] = float(row['score'])
        if not math.isfinite(row['score']):
            raise ValueError('Nonfinite fixed-baseline margin')
    result = receipt['result']
    counted = audit({'dear_r': rows}, {'dear_r': result})['dear_r']
    rates = {k: Fraction(v['exact_ba']) for k, v in counted['metrics'].items()}
    pairs = {k: Fraction(v['exact_ba_drop']) for k, v in result['pairs'].items()}
    summaries = {
        'minimum_domain_ba': min(v for k, v in rates.items() if k.split('/')[1] == 'all'),
        'minimum_scene_ba': min(v for k, v in rates.items() if k.split('/')[1] != 'all'),
        'maximum_domain_drop': max(v for k, v in pairs.items() if k.split('/')[1] == 'all'),
        'maximum_any_scene_drop': max(pairs.values()),
    }
    if any(abs(float(v) - result[k]) > 1e-15 for k, v in summaries.items()):
        raise ValueError('Fixed-baseline extrema differ')
    bad = json.loads(json.dumps(result))
    bad['metrics'][next(iter(bad['metrics']))]['balanced_accuracy_at_zero'] += .01
    try:
        audit({'dear_r': rows}, {'dear_r': bad})
    except ValueError as error:
        if 'confusion count' not in str(error):
            raise
    else:
        raise ValueError('Perturbed BA was not rejected')
    value = {
        'passed': True, 'images': len(inventory), 'records': len(rows),
        'metric_groups': counted['metric_groups'], 'paired_comparisons': counted['paired_comparisons'],
        'integer_counts': counted['metrics'], 'exact_extrema': {k: str(v) for k, v in summaries.items()},
        'perturbed_ba_rejected': True,
        'iteration_sha256': file_sha256(OUTPUT / 'iteration.json'),
        'scores_sha256': file_sha256(OUTPUT / 'scores.csv'),
        'script_sha256': file_sha256(Path(__file__)),
        'scope': 'Coverage and shared-score arithmetic only; not independent scientific validation',
    }
    destination.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: value[k] for k in ('passed', 'images', 'records', 'metric_groups', 'paired_comparisons')}))


if __name__ == '__main__':
    main()
