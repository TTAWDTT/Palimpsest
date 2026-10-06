"""Verify original bytes and screen decoded RGB/pHash across development roles."""

from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from struct import pack
from time import perf_counter

from PIL import Image

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.data_preparation.chimera.audit_content_pairs import phash
from experiments.data_preparation.development_roles.audit_roles import audit_inventory
from experiments.origin_detection.source_view_risk.run_iteration import parent_data
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json

OUTPUT=WORK_DIR/'robust_statistics/development_content'


def fingerprint(path):
    with Image.open(path) as image:
        rgb=image.convert('RGB')
        digest=sha256(pack('<II',rgb.width,rgb.height)+rgb.tobytes()).hexdigest()
        width,height=rgb.size
    return {'decoded_rgb_sha256':digest,'width':width,'height':height,'phash64':f'{phash(path):016x}'}


def summarize(records):
    seen=set();digests=defaultdict(list)
    for r in records:
        key=r['domain'],r['src']
        if key in seen or r['role'] not in {'fit','threshold','selection'} or r['label'] not in {'REAL','FAKE'}:
            raise ValueError('Invalid/duplicate original source')
        seen.add(key);digests[r['decoded_rgb_sha256']].append(r)
    exact=[{'digest':k,'records':v,'cross_role':len({r['role'] for r in v})>1,
            'conflicting_labels':len({r['label'] for r in v})>1} for k,v in sorted(digests.items()) if len(v)>1]
    candidates=[]
    for i,j in combinations(range(len(records)),2):
        a,b=records[i],records[j]
        distance=(int(a['phash64'],16)^int(b['phash64'],16)).bit_count()
        if distance<=6:
            candidates.append({'indices':[i,j],'hamming':distance,'cross_role':a['role']!=b['role'],
                               'conflicting_labels':a['label']!=b['label'],
                               'exact_rgb':a['decoded_rgb_sha256']==b['decoded_rgb_sha256']})
    return {'originals':len(records),'exact_rgb_groups':exact,'exact_rgb_group_count':len(exact),
        'exact_rgb_cross_role_groups':sum(g['cross_role'] for g in exact),
        'exact_rgb_conflicting_label_groups':sum(g['conflicting_labels'] for g in exact),
        'phash_candidate_pairs':sorted(candidates,key=lambda x:(x['hamming'],x['indices'])),
        'phash_candidate_count':len(candidates),'phash_cross_role_pairs':sum(g['cross_role'] for g in candidates),
        'phash_conflicting_label_pairs':sum(g['conflicting_labels'] for g in candidates)}


def code_pins():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    paths+=[REPO_ROOT/p for p in ('tests/evaluation/test_development_content.py',
        'experiments/data_preparation/chimera/audit_content_pairs.py',
        'experiments/data_preparation/development_roles/audit_roles.py',
        'experiments/origin_detection/source_view_risk/run_iteration.py',
        'experiments/origin_detection/phase_statistics/run_iteration.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)}


def verify_counts(result,expected):
    for key,value in expected.items():
        if result[key]!=value:raise ValueError('Content count claim differs')


def main():
    if (OUTPUT/'audit.json').exists():raise FileExistsError('Preserve original content audit')
    controls=json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins']!=code_pins():raise ValueError('Content controls changed')
    _,inventory,parent,_=parent_data()
    structural=audit_inventory(inventory)
    if structural['native_records']!=6300 or structural['sources']!=2100:
        raise ValueError('Development source panel changed')
    selected=sorted([r for r in inventory if r['condition']=='original'],key=lambda r:(r['domain'],r['src']))
    records=[];start=perf_counter()
    for row in selected:
        path=image_path(row)
        if file_sha256(path)!=row['sha256']:raise ValueError('Original encoded bytes changed')
        value=fingerprint(path)
        if value['width']!=int(row['width']) or value['height']!=int(row['height']):
            raise ValueError('Original decoded dimensions changed')
        records.append({**{k:row[k] for k in ('domain','src','scene','role','label','filename','sha256')},**value})
        if len(records)%100==0:print(json.dumps({'originals':len(records),'elapsed_s':perf_counter()-start}),flush=True)
    result=summarize(records)
    write_json(OUTPUT/'audit.json',{**result,'records':records,'code_pins':code_pins(),
        'inventory_sha256':parent['inventory_sha256'],'elapsed_s':perf_counter()-start,
        'scope':'Original-only decoded RGB identity and pHash candidates;not certified near duplicates or independence'})
    print(json.dumps({k:v for k,v in result.items() if k not in ('exact_rgb_groups','phash_candidate_pairs')}),flush=True)


if __name__=='__main__':main()
