"""Post-fit diagnostic of all variant/scene pairs; never refits a threshold."""

from collections import defaultdict
import json
from fractions import Fraction

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.evaluation.classification import evaluate
from palimpsest.evaluation.pairing import paired_change
from palimpsest.io.tables import read_rows
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.residual_training.fit_rules import TRAINING_VARIANT
from experiments.origin_detection.paired_stability.run_iteration import INTERVENTION

OUTPUT=WORK_DIR/'robust_statistics/frozen_clip'


def exact_ba(records):
    rates=[]
    for label in ('REAL','FAKE'):
        selected=[r for r in records if r['label']==label]
        if not selected: raise ValueError('Missing class')
        correct=sum((float(r['score'])>0)==(label=='FAKE') for r in selected)
        rates.append(Fraction(correct,len(selected)))
    return sum(rates)/2


def main():
    destination=OUTPUT/'paired_conditions_audit.json'
    if destination.exists(): raise FileExistsError(destination)
    receipt=json.loads((OUTPUT/'features.json').read_text(encoding='utf-8'))
    iteration=json.loads((OUTPUT/'iteration.json').read_text(encoding='utf-8'))
    path=OUTPUT/'clip_both_rule.json'
    if file_sha256(OUTPUT/'features.csv')!=receipt['csv_sha256'] or file_sha256(path)!=iteration['rule_files']['clip/both']:
        raise ValueError('Feature/rule changed')
    rule=StableRule.load(path)
    rows=[r for r in read_rows(OUTPUT/'features.csv') if r['role']=='selection']
    if len(rows)!=3780: raise ValueError('Selection denominator differs')
    scores=rule.score([[float(r[n]) for n in FEATURE_NAMES] for r in rows])-rule.threshold
    groups=defaultdict(list)
    for row,score in zip(rows,scores):
        value={**row,'score':float(score)}
        groups[(row['domain'],row['scene'],row['condition'],row['variant'])].append(value)
        if row['scene']!='all': groups[(row['domain'],'all',row['condition'],row['variant'])].append(value)
    metrics={'/'.join(key):evaluate(records,'score') for key,records in groups.items()}
    pairs={}
    for domain,scene,condition,variant in sorted(groups):
        if condition!='original':
            ref=(domain,scene,'original',variant)
            if ref not in groups: raise ValueError('Missing original pair')
            name='/'.join((domain,scene,'original>'+condition,variant))
            original,processed=groups[ref],groups[(domain,scene,condition,variant)]
            result=paired_change({r['src']:r for r in original},{r['src']:r for r in processed})
            result['exact_ba_drop']=str(exact_ba(original)-exact_ba(processed)); pairs[name]=result
    variants=('raw',TRAINING_VARIANT,INTERVENTION)
    if {key[3] for key in groups}!=set(variants): raise ValueError('Diagnostic variant set differs')
    # Lock comparisons before arithmetic; Q70 is exposed diagnostic, not new holdout.
    for domain,scene,condition in sorted({key[:3] for key in groups}):
        for before,after in ((variants[0],variants[1]),(variants[0],variants[2]),(variants[1],variants[2])):
            original,processed=groups[(domain,scene,condition,before)],groups[(domain,scene,condition,after)]
            name='/'.join((domain,scene,condition,before+'>'+after))
            result=paired_change({r['src']:r for r in original},{r['src']:r for r in processed})
            result['exact_ba_drop']=str(exact_ba(original)-exact_ba(processed)); pairs[name]=result
    all_pairs={k:v for k,v in pairs.items() if k.split('/')[1]=='all'}
    scene_pairs={k:v for k,v in pairs.items() if k.split('/')[1]!='all'}
    worst=max(pairs,key=lambda k:Fraction(pairs[k]['exact_ba_drop']))
    write_json(destination,{'metrics':metrics,'pairs':pairs,'maximum_domain_condition_drop':float(max(Fraction(v['exact_ba_drop']) for v in all_pairs.values())),
                           'maximum_scene_drop':float(max(Fraction(v['exact_ba_drop']) for v in scene_pairs.values())),
                           'worst_pair':worst,'minimum_scene_ba':min(v['balanced_accuracy_at_zero'] for k,v in metrics.items() if k.split('/')[1]!='all'),
                           'rule_sha256':file_sha256(path),'iteration_sha256':file_sha256(OUTPUT/'iteration.json'),
                           'features_receipt_sha256':file_sha256(OUTPUT/'features.json'),
                           'scope':'Postfit explanatory diagnostic including all encoding reference states and scenes;no recalibration/new independent data'})
    print(json.dumps({'groups':len(groups),'pairs':len(pairs),'worst':worst,'drop':pairs[worst]['exact_ba_drop']}))


if __name__=='__main__':main()
