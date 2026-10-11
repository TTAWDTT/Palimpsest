"""Audit source roles and exact cached byte identities without decoding pixels."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.source_view_risk.run_iteration import parent_data

OUTPUT = WORK_DIR / 'robust_statistics/development_roles'
ROLES = {'fit', 'threshold', 'selection'}
LABELS = {'REAL', 'FAKE'}
CONDITIONS = {'rr': {'original', 'transfer', 'redigital'},
              'chimera': {'original', 'mac_iphone', 'lg_blackfly'}}


def audit_inventory(rows):
    """Count identities and role overlaps; preserve detected duplicate groups."""
    sources = defaultdict(list)
    seen = set()
    for row in rows:
        domain, src, condition = (row[k] for k in ('domain', 'src', 'condition'))
        identity = domain, src, condition
        if identity in seen:
            raise ValueError('Duplicate native record identity')
        seen.add(identity)
        if (domain not in CONDITIONS or condition not in CONDITIONS[domain]
                or row['role'] not in ROLES or row['label'] not in LABELS):
            raise ValueError('Unknown domain, condition, role or label')
        if not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid cached byte digest')
        if int(row['width']) <= 0 or int(row['height']) <= 0:
            raise ValueError('Nonpositive native dimensions')
        sources[domain, src].append(row)
    for (domain, _), selected in sources.items():
        if ({r['condition'] for r in selected} != CONDITIONS[domain]
                or len({(r['role'], r['label'], r['scene']) for r in selected}) != 1):
            raise ValueError('Incomplete or conflicting source panel')
    result = {'native_records': len(rows), 'sources': len(sources),
              'source_role_counts': dict(Counter(v[0]['role'] for v in sources.values())),
              'source_domain_role_counts': dict(Counter('/'.join((k[0], v[0]['role']))
                                                      for k, v in sources.items()))}
    for scope, selected in (('originals', [r for r in rows if r['condition'] == 'original']),
                            ('all_native', rows)):
        digests = defaultdict(list)
        for row in selected:
            digests[row['sha256']].append({k: row[k] for k in
                ('domain', 'src', 'condition', 'role', 'label', 'scene', 'filename')})
        duplicates = []
        for digest, records in sorted(digests.items()):
            if len(records) < 2:
                continue
            duplicates.append({'sha256': digest, 'records': records,
                'distinct_sources': len({(r['domain'], r['src']) for r in records}),
                'cross_role': len({r['role'] for r in records}) > 1,
                'conflicting_labels': len({r['label'] for r in records}) > 1})
        result[scope] = {'duplicate_groups': duplicates,
            'multi_record_groups': len(duplicates),
            'distinct_source_groups': sum(g['distinct_sources'] > 1 for g in duplicates),
            'cross_role_groups': sum(g['cross_role'] for g in duplicates),
            'conflicting_label_groups': sum(g['conflicting_labels'] for g in duplicates),
            'within_source_only_groups': sum(g['distinct_sources'] == 1 for g in duplicates)}
    return result


def code_pins():
    files = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md',
        REPO_ROOT / 'tests/evaluation/test_development_roles.py',
        REPO_ROOT / 'experiments/origin_detection/source_view_risk/run_iteration.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def main():
    if (OUTPUT / 'audit.json').exists():
        raise FileExistsError('Preserve signed role audit')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text(encoding='utf-8'))
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Role audit artificial controls changed')
    _, inventory, receipt, iteration = parent_data()
    audit = audit_inventory(inventory)
    if (audit['native_records'] != 6300 or audit['sources'] != 2100
            or audit['source_role_counts'] != {'fit': 1260, 'threshold': 420, 'selection': 420}
            or receipt['inventory_sha256'] != iteration['inventory_sha256']):
        raise ValueError('Signed development source totals changed')
    audit.update({'metadata_structure_passed': True, 'code_pins': code_pins(),
        'inventory_sha256': receipt['inventory_sha256'],
        'parent_features_receipt_sha256': file_sha256(WORK_DIR / 'robust_statistics/frozen_clip/features.json'),
        'scope': 'Existing cached byte hashes only;no decoded/near-duplicate or independence certification'})
    (OUTPUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k in
        ('metadata_structure_passed', 'native_records', 'sources', 'source_role_counts')}))
    for scope in ('originals', 'all_native'):
        print(json.dumps({scope: {k: v for k, v in audit[scope].items() if k != 'duplicate_groups'}}))


if __name__ == '__main__':
    main()
