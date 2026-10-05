"""Pre-real-data controls: independent orbit enumeration, pixels and tree truth."""

import csv
from dataclasses import replace
from itertools import product
import json
import tomllib

import numpy as np
from PIL import Image
import pytest
from sklearn.ensemble import RandomForestClassifier

from palimpsest.detection.algorithms.forest import FeatureTree, ForestRule
from palimpsest.detection.algorithms.residual_statistics.features import (
    ORBITS, FEATURE_NAMES, quantize, triplet_counts, residual_histogram, extract_features,
)
from palimpsest.detection.algorithms.residual_statistics.detector import ResidualDetector
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from experiments.origin_detection.residual_statistics import run_iteration as runner


def independent_orbits():
    remaining = set(product(range(-2,3),repeat=3))
    groups = []
    while remaining:
        a,b,c = min(remaining)
        group = {(a,b,c),(c,b,a),(-a,-b,-c),(-c,-b,-a)}
        groups.append(group)
        remaining -= group
    return groups


def slow_counts(q,axis):
    groups=independent_orbits()
    output=np.zeros(39,int)
    for y in range(q.shape[0]-(2 if axis==0 else 0)):
        for x in range(q.shape[1]-(2 if axis==1 else 0)):
            triple=tuple(int(q[y+j*(axis==0),x+j*(axis==1)]) for j in range(3))
            output[next(i for i,g in enumerate(groups) if triple in g)]+=1
    return output


def test_orbits_neighbor_counts_and_broken_adjacency():
    groups=independent_orbits()
    assert len(groups)==39 and sum(map(len,groups))==125
    assert tuple(min(g) for g in groups)==ORBITS
    q=np.array([[-2,0,2,-2,0,2],[1,-1,2,0,1,-2],[2,2,-1,0,-2,1]],np.int8)
    for axis in (0,1):
        expected=slow_counts(q,axis)
        np.testing.assert_array_equal(triplet_counts(q,axis),expected)
        np.testing.assert_array_equal(triplet_counts(-q,axis),expected)
        np.testing.assert_array_equal(triplet_counts(np.flip(q,axis),axis),expected)
    altered=q[:,[0,2,1,3,5,4]]
    assert not np.array_equal(triplet_counts(altered,1),triplet_counts(q,1))


def test_second_difference_null_quadratic_and_symmetry():
    y,x=np.mgrid[:31,:37]
    zero=ORBITS.index((0,0,0));expected=np.zeros(39);expected[zero]=1
    np.testing.assert_array_equal(residual_histogram(7*y+3*x+19),expected)
    quad=np.zeros(39);quad[ORBITS.index((-1,-1,-1))]=1
    np.testing.assert_array_equal(residual_histogram(x*x+y*y),quad)
    pattern=((17*x+23*y+3*x*y)%251).astype(np.float32)
    ref=residual_histogram(pattern)
    np.testing.assert_array_equal(residual_histogram(pattern+11),ref)
    for alternate in (pattern.T,np.rot90(pattern),255-pattern):
        np.testing.assert_allclose(residual_histogram(alternate),ref,atol=1e-12,rtol=0)


def test_quantization_and_gain_floor_counterexample():
    for value,code in [(0,0),(1,1),(3,2),(6,2),(-1,-1),(-3,-2),(.2,0),(.4,1)]:
        np.testing.assert_array_equal(quantize(np.full((9,11),value)),np.full((9,11),code))
    # Fixed +1 normalization floor explicitly prevents exact gain invariance.
    assert not np.array_equal(quantize(np.full((9,11),.2)),quantize(np.full((9,11),.4)))
    for bad in (np.full((9,11),np.nan),np.zeros(5),np.empty((0,2))):
        with pytest.raises(ValueError):quantize(bad)
    with pytest.raises(ValueError):triplet_counts(np.full((3,3),3,np.int8),1)
    with pytest.raises(ValueError):triplet_counts(np.zeros((3,3)),1)


def test_independent_rgb_block_order_and_pyramid_answers():
    y,x=np.mgrid[:32,:32]
    rgb=np.stack([np.full_like(x,128),(x%2)*255,((x+y)%2)*255],axis=-1).astype(np.uint8)
    blocks=extract_features(rgb).values[28:].reshape(9,39)
    zero=ORBITS.index((0,0,0));alternating=ORBITS.index((-2,2,-2))
    expected=np.zeros((9,39));expected[:,zero]=1
    expected[3,zero]=.5;expected[3,alternating]=.5
    expected[6,zero]=0;expected[6,alternating]=1
    np.testing.assert_array_equal(blocks,expected)
    # Swapping channels is deliberately detectable, not silently pooled.
    assert not np.array_equal(extract_features(rgb[...,::-1]).values[28:],blocks.ravel())


@pytest.mark.parametrize('shape',[(1,1,3),(17,8,3),(256,256,3)])
def test_constant_pixel_histograms_and_shared_ordinal(shape):
    result=extract_features(np.full(shape,128,np.uint8))
    assert result.values.shape==(379,)
    expected=np.zeros(28)
    expected.reshape(4,7)[:,5:]=1
    np.testing.assert_array_equal(result.values[:28],expected)
    blocks=result.values[28:].reshape(9,39)
    np.testing.assert_array_equal(blocks.sum(axis=1),np.ones(9))
    assert np.all(blocks[:,ORBITS.index((0,0,0))]==1)
    np.testing.assert_array_equal(extract_features(np.full(shape,128,np.uint8),residual=False).values,expected)
    with pytest.raises(ValueError):extract_features(np.zeros(shape,dtype=float))


def test_manual_tree_cut_equality_precision_and_schema(tmp_path):
    tree=FeatureTree((1,-1,-1),(2,-1,-1),(0,-2,-2),(.25,-2.,-2.),(.5,0.,1.))
    rule=ForestRule(('x',),(tree,))
    values=np.array([[.0],[.25],[.25+1e-10],[np.nextafter(np.float32(.25),np.float32(1.0))]])
    np.testing.assert_array_equal(rule.score(values),[-.5,-.5,-.5,.5])
    path=tmp_path/'tree.json';rule.save(path)
    restored=ForestRule.load(path)
    assert restored.fingerprint==rule.fingerprint
    np.testing.assert_array_equal(restored.score(values),rule.score(values))
    payload=json.loads(path.read_text(encoding='utf-8'));payload['input_precision']='float64'
    path.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError):ForestRule.load(path)
    for bad in (replace(tree,left=(0,-1,-1)),replace(tree,right=(1,-1,-1)),
                replace(tree,probability_ai=(.5,-.1,1.)),replace(tree,feature=(2,-2,-2)),
                replace(tree,left=(1.0,-1,-1))):
        with pytest.raises(ValueError):ForestRule(('x',),(bad,))
    with pytest.raises(ValueError):rule.score([[np.nan]])


def test_portable_vs_sklearn_random_and_boundaries():
    rng=np.random.default_rng(42);x=rng.uniform(-1,1,(512,7));labels=(x[:,0]*x[:,1]>0).astype(int)
    estimator=RandomForestClassifier(n_estimators=17,max_depth=6,min_samples_leaf=8,random_state=42).fit(x,labels)
    rule=ForestRule.from_estimator(estimator,tuple(str(i) for i in range(7)))
    probes=rng.uniform(-1,1,(256,7))
    for tree in estimator.estimators_:
        f=tree.tree_.feature[0];cut=tree.tree_.threshold[0]
        for v in (cut,np.nextafter(np.float32(cut),np.float32(np.inf)),cut+1e-10):
            p=np.zeros(7);p[f]=v;probes=np.vstack([probes,p])
    np.testing.assert_allclose(rule.score(probes),estimator.predict_proba(probes)[:,1]-.5,atol=1e-12,rtol=0)


def classification_plant():
    rows=[]
    for domain,scenes,conditions in [('rr',['all'],['original','transfer','redigital']),
                                    ('chimera',['cat','church','horse'],['original','mac_iphone','lg_blackfly'])]:
        for scene in scenes:
            for role in ('fit','threshold','selection'):
                for i in range(32):
                    a,b=(i//8)//2,(i//8)%2;src=f'{domain}/{scene}/{role}/{i}'
                    base=np.zeros(379)
                    for start in range(0,28,7):base[start+a]=1
                    for start in range(28,379,39):base[start+b]=1
                    for c in conditions:
                        for variant in runner.VARIANTS:
                            rows.append({'filename':src+'/'+c,'src':src,'domain':domain,'scene':scene,'role':role,
                                         'condition':c,'label':'FAKE' if a!=b else 'REAL','sha256':'plant','variant':variant,
                                         **dict(zip(FEATURE_NAMES,base.tolist()))})
    return rows


def test_csv_classifier_plant_and_null_before_real_data(tmp_path):
    rows=classification_plant();inventory=[r for r in rows if r['variant']=='raw']
    path=tmp_path/'plant.csv'
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    clean=runner.audit_cache(read_rows(path),inventory)
    with runner.CONFIG.open('rb') as stream:config=tomllib.load(stream)
    result,rules=runner.run_screen(clean,config,'plant')
    assert result['candidates']['hybrid']['gate_passed']
    assert not result['candidates']['ordinal256']['gate_passed']  # Missing second XOR coordinate.
    for v in result['candidates']['hybrid']['selection_views'].values():
        assert v['auc']==1 and v['balanced_accuracy_at_zero']==1
    zero=[{**r,**dict.fromkeys(FEATURE_NAMES,0.0)} for r in rows]
    null=runner.screen_rule(zero,FEATURE_NAMES,rules['hybrid'],config)
    assert all(v['auc']==.5 for v in null['selection_views'].values())
    assert not null['gate_passed']
    for mutation in ('duplicate','missing','label','role','nan','sum'):
        broken=[dict(r) for r in rows]
        if mutation=='duplicate':broken.append(dict(broken[0]))
        if mutation=='missing':broken.pop()
        if mutation=='label':broken[0]['label']='FAKE'
        if mutation=='role':broken[0]['role']='selection'
        if mutation=='nan':broken[0][FEATURE_NAMES[-1]]=float('nan')
        if mutation=='sum':broken[0][FEATURE_NAMES[-1]]=.2
        with pytest.raises(ValueError):runner.audit_cache(broken,inventory)


def test_file_extraction_identity_and_live_score(tmp_path,monkeypatch):
    path=tmp_path/'image.png';Image.fromarray(np.full((32,32,3),128,np.uint8)).save(path)
    row={'filename':'plant.png','src':'plant','domain':'rr','scene':'all','condition':'original',
         'label':'REAL','role':'fit','sha256':file_sha256(path),'width':32,'height':32}
    monkeypatch.setattr(runner,'image_path',lambda _:path)
    directory=tmp_path/'clean';directory.mkdir()
    result=runner.extract_inventory([row],directory)
    assert result['images']==1 and result['records']==2
    for bad in ({**row,'sha256':'bad'},{**row,'width':31}):
        with pytest.raises(ValueError):runner.extract_inventory([bad],directory)
    tree=FeatureTree((-1,),(-1,),(-2,),(-2.,),(.5,))
    detector=ResidualDetector(ForestRule(FEATURE_NAMES,(tree,)))
    items=[{'filename':'plant.png','path':path,'sha256':file_sha256(path),'expected_score':0.0}]
    result=benchmark_files(detector,items,repeats=1)
    assert result['summaries']['all']['images']==1
    with pytest.raises(ValueError,match='prediction mismatch'):
        benchmark_files(detector,[{**items[0],'expected_score':.1}],repeats=1)
