"""One-off metadata cost plan for the already sealed800-source ZIP selection.

Never fetches image bytes,changes the cohort or binds/runs a detector. Uses next
indexed local offset to cover complete entries without guessing JPEG headers.
"""

import argparse
import json
from pathlib import Path

from palimpsest.io.range_plan import coalesce_ranges
from palimpsest.io.hashing import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True); parser.add_argument('--sealed-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve range cost plan')
    plan = json.loads(args.sealed_plan.read_text()); inventory = json.loads(args.inventory.read_text())
    if file_sha256(args.inventory) != plan['inventory_sha256']: raise ValueError('ZIP metadata changed')
    if plan['source_count'] != 800 or plan['image_count'] != 3200: raise ValueError('Sealed cohort changed')
    ordered = sorted(inventory, key=lambda r: r['local_offset'])
    offsets = [r['local_offset'] for r in ordered]
    if len(set(offsets)) != len(offsets) or min(offsets) < 0 or max(offsets) >= plan['archive_bytes']:
        raise ValueError('ZIP offset identity/bounds differ')
    full = {r['name']: r for r in ordered}
    end = {r['name']: offsets[i+1]-1 if i+1 < len(ordered) else plan['archive_bytes']-1
           for i, r in enumerate(ordered)}
    spans = []
    for row in plan['entries']:
        if full.get(row['name']) != {k: row[k] for k in full[row['name']]}:
            raise ValueError('Selected ZIP entry differs')
        start, stop = row['local_offset'], end[row['name']]
        if stop-start+1 < row['compressed_bytes']+30: raise ValueError('ZIP entry span cannot fit payload/header')
        spans.append((start, stop))
    payload = sum(r['compressed_bytes'] for r in plan['entries']); entries_bytes = sum(b-a+1 for a, b in spans)
    results = []
    for gap in (0, 65536, 262144, 1048576, 4194304, 16777216):
        merged = coalesce_ranges(spans, maximum_gap=gap, maximum_span=67108864)
        total = sum(b-a+1 for a, b in merged)
        # Verify interval containment independently without expanding archive bytes.
        cursor = 0
        for start, stop in sorted(spans):
            while merged[cursor][1] < start: cursor += 1
            if not merged[cursor][0] <= start <= stop <= merged[cursor][1]: raise ValueError('Coalescing lost selected bytes')
        results.append({'maximum_gap_bytes': gap, 'maximum_span_bytes': 67108864, 'requests': len(merged),
            'range_bytes': total, 'extra_gap_bytes': total-entries_bytes,
            'minimum_interrequest_delay_s_at2s': 2*max(0, len(merged)-1), 'spans': merged})
    result = {'source_count': 800, 'image_count': 3200, 'compressed_payload_bytes': payload,
        'complete_indexed_entry_bytes': entries_bytes, 'sealed_plan_sha256': file_sha256(args.sealed_plan),
        'source_list_sha256': plan['source_list_sha256'], 'inventory_sha256': file_sha256(args.inventory),
        'script_sha256': file_sha256(Path(__file__)), 'coalescer_sha256': file_sha256(Path('src/palimpsest/io/range_plan.py')),
        'choices': results, 'pixels_fetched': 0, 'classifier_bound': False,
        'scope': 'Indexed ZIP span cost only;not actual fetch ETA or pixel/integrity/independence evidence',
        'limits': ['Next indexed local offset assumed entry boundary;real downloader must parse local headers and validate payload CRC/length',
                   'Last entry span ends at archive size and can include central directory',
                   'Gap traffic includes unselected compressed entries;do not decode or evaluate them',
                   '2sdelay is historical planning assumption;server headers/retries/transfer still need measurement']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'spans'} for r in results]))


if __name__ == '__main__': main()
