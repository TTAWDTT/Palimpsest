"""Source-consistent, truth-orthogonal sham targets for a negative control.

True labels enter only this control's assignment. This is a constrained sham
experiment, not an ordinary permutation test or a deployment algorithm.
"""

from collections import defaultdict
import hashlib
import json
import random


def balanced_source_null(records, *, seed):
    """Return sham labels keyed by (domain,src), with exact within-scene balance.

    Refuse non-fit records, inconsistent sources, absent classes, or odd true
    class counts. Every true class contributes half sham0 and half sham1;
    repeated image views do not affect the assignment.
    """
    sources = {}
    for row in records:
        if row['role'] != 'fit' or row['label'] not in ('REAL', 'FAKE'):
            raise ValueError('Balanced control requires fit-only binary labels')
        key = (row['domain'], row['src'])
        value = (row['scene'], int(row['label'] == 'FAKE'))
        if key in sources and sources[key] != value:
            raise ValueError('Conflicting source metadata in balanced control')
        sources[key] = value
    if not sources:
        raise ValueError('Empty balanced control')
    strata = defaultdict(list)
    for key, (scene, truth) in sorted(sources.items()):
        strata[(key[0], scene, truth)].append(key)
    cells = {(domain, scene) for domain, scene, _ in strata}
    for domain, scene in cells:
        for truth in (0, 1):
            count = len(strata[(domain, scene, truth)])
            if count == 0 or count % 2:
                raise ValueError('Balanced control requires even nonempty true classes')
    rng = random.Random(seed)
    labels, counts = {}, []
    for (domain, scene, truth), keys in sorted(strata.items()):
        values = [0] * (len(keys) // 2) + [1] * (len(keys) // 2)
        rng.shuffle(values)
        labels.update(zip(keys, values))
        counts.append({'domain': domain, 'scene': scene, 'truth': truth,
                       'sham_zero': len(keys) // 2, 'sham_one': len(keys) // 2})
    assignment = [{'domain': key[0], 'src': key[1], 'scene': sources[key][0],
                   'truth': sources[key][1], 'sham': labels[key]} for key in sorted(labels)]
    encoded = json.dumps(assignment, sort_keys=True, separators=(',', ':')).encode()
    return labels, {'seed': seed, 'sources': len(labels), 'cells': counts,
                    'assignment': assignment, 'assignment_sha256': hashlib.sha256(encoded).hexdigest(),
                    'scope': 'Exact fit truth/sham independence;not an independent data validation'}
