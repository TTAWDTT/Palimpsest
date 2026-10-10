"""Exact printed-subset mean intervals with planted wrong-average refusal."""

from fractions import Fraction
import json
from pathlib import Path

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR


def compare_mean(values,reported):
    values=[Fraction(str(v)) for v in values]
    if not values:raise ValueError('Empty printed table')
    mean=sum(values)/len(values);quoted=Fraction(str(reported));half=Fraction(1,20)
    overlap=max(mean-half,quoted-half)<=min(mean+half,quoted+half)
    return {'exact_subset_mean':str(mean),'subset_mean_percent':float(mean),
            'rounded_input_mean_interval':[float(mean-half),float(mean+half)],
            'reported_mean_interval':[float(quoted-half),float(quoted+half)],
            'rounding_can_explain':overlap}


def main():
    root=WORK_DIR/'robust_statistics/spectral_tail_review';path=root/'table_audit.json'
    if path.exists():raise FileExistsError('Preserve primary table audit')
    positive=compare_mean(['80.0','100.0'],'90.0');negative=compare_mean(['80.0','100.0'],'91.0')
    if not positive['rounding_can_explain'] or negative['rounding_can_explain']:
        raise ValueError('Printed mean known controls failed')
    controls={'positive':positive,'wrong_mean':negative,'wrong_mean_rejected':True,'script_sha256':file_sha256(Path(__file__))}
    (root/'table_controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    stal=['97.7','96.9','95.9','70.8','97.8','96.8','95.3','97.0','84.4','81.7','95.9','96.7','96.4']
    aide=['63.4','48.8','51.9']
    results={'table9_stal_own92.6':compare_mean(stal,'92.6'),
             'table9_stal_vs_table1_94.7':compare_mean(stal,'94.7'),
             'table5_aide_own54.7':compare_mean(aide,'54.7'),
             'table5_aide_vs_table1_48.8':compare_mean(aide,'48.8')}
    path.write_text(json.dumps({'results':results,'table9_stal_values':stal,'table5_aide_values':aide,
        'pdf_sha256':file_sha256(root/'main.pdf'),'controls_sha256':file_sha256(root/'table_controls.json'),
        'scope':'Printed unweighted mean arithmetic only;unresolved checkpoint/subset definitions;not method reproduction'},indent=2)+'\n')
    print(json.dumps(results))


if __name__=='__main__':main()
