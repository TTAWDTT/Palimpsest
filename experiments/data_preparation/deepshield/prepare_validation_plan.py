"""Metadata-only sealed800-source plan; no image bytes or model outputs used."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path,PurePosixPath

from palimpsest.io.hashing import file_sha256

CONDITIONS=('PreSocial','Facebook','Telegram','X')
EXPOSED=('Real/FFHQ/00741','Fake/FLUX.1/general/00098')
SEED=20261006


def source_key(name):
    parts=PurePosixPath(name).parts
    if len(parts)<5 or parts[0]!='DeepShield' or parts[1] not in CONDITIONS:return None
    return '/'.join((*parts[2:-1],PurePosixPath(parts[-1]).stem))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve sealed validation plan')
    record=json.loads(args.record.read_text(encoding='utf-8'))
    archive=[f for f in record['files'] if f['key']=='DeepShield_Dataset.zip']
    if len(archive)!=1 or archive[0]['size']!=167573270584:raise ValueError('Published archive size changed')
    inventory=json.loads(args.inventory.read_text(encoding='utf-8'));paired=defaultdict(dict)
    for entry in inventory:
        name=entry['name'];key=source_key(name)
        if key is None or not entry['uncompressed_bytes'] or PurePosixPath(name).suffix.lower() not in ('.jpg','.jpeg','.png'):continue
        condition=PurePosixPath(name).parts[1]
        if condition in paired[key]:raise ValueError('Duplicate source/condition in archive index')
        paired[key][condition]=entry
    strata=defaultdict(list)
    for key,views in paired.items():
        if set(views)==set(CONDITIONS) and key not in EXPOSED:
            strata[str(PurePosixPath(key).parent)].append(key)
    def ordered(keys):return sorted(keys,key=lambda k:(sha256(f'{SEED}/{k}'.encode()).hexdigest(),k))
    selected=[];quotas={}
    for name in sorted(strata):
        if name.startswith('Real/'):
            if name not in ('Real/FFHQ','Real/FORLAB'):raise ValueError('New real stratum requires review')
            quotas[name]=200
    fake_models=sorted({PurePosixPath(k).parts[1] for k in strata if k.startswith('Fake/')})
    if len(fake_models)!=8:raise ValueError('Fake generator count differs')
    for model in fake_models:
        parts=sorted(k for k in strata if k.startswith('Fake/'+model+'/'))
        quotient,remainder=divmod(50,len(parts))
        for index,name in enumerate(parts):quotas[name]=quotient+(index<remainder)
    for stratum,count in sorted(quotas.items()):
        pool=ordered(strata[stratum])
        if len(pool)<count:raise ValueError('Insufficient sealed paired stratum')
        selected.extend(pool[:count])
    if len(selected)!=800 or len(set(selected))!=800:raise ValueError('Sealed source count differs')
    rows=[]
    for key in sorted(selected):
        for condition in CONDITIONS:
            rows.append({'source':key,'label':'REAL' if key.startswith('Real/') else 'FAKE',
                         'condition':'original' if condition=='PreSocial' else condition.lower(),
                         **paired[key][condition]})
    payload={
        'schema':1,'source_count':800,'image_count':3200,'class_source_counts':{'REAL':400,'FAKE':400},
        'quotas':quotas,'excluded_exposed_sources':EXPOSED,'selection':'Seeded SHA order within fixed generator/content strata',
        'seed':SEED,'inventory_sha256':file_sha256(args.inventory),'record_sha256':file_sha256(args.record),
        'archive_bytes':archive[0]['size'],'published_archive_checksum':archive[0]['checksum'],
        'full_archive_md5_verified':False,'pixels_downloaded_by_this_program':0,'classifier_bound':False,
        'compressed_payload_bytes':sum(r['compressed_bytes'] for r in rows),
        'range_request_upper_plan_one_per_image':len(rows),
        'range_bytes_plus_1024_header_allowance':sum(r['compressed_bytes'] for r in rows)+1024*len(rows),
        'source_list_sha256':sha256(json.dumps(sorted(selected),separators=(',',':')).encode()).hexdigest(),
        'role':'external_validation_sealed_pending_integrity_overlap_and_fixed_method',
        'limitations':['Metadata mapping is not per-file pixel certification','Model/pretraining/near-duplicate overlap not audited',
                      'True digital platform comparisons only,no physical capture','Do not tune after scores;bind method before inference',
                      '800sources is a cost plan,not assured2pp confidence coverage'],
        'entries':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('entries','quotas')}))


if __name__=='__main__':main()
