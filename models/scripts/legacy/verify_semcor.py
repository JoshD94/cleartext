import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from collections import defaultdict
import numpy as np
from cleartext.data import ROOT
from cleartext.wsd import SenseModel
run=ROOT/'runs/semcor-20260924'
data,manifest=pickle.load(open(ROOT/'data/cache/semcor-records-v1.pkl','rb'))
for a,b in [('train','dev'),('train','test'),('dev','test')]:
    assert not set(manifest['documents'][a])&set(manifest['documents'][b])
    assert not {Path(f).name for f in manifest['documents'][a]}&{Path(f).name for f in manifest['documents'][b]}
    assert not {r['sentence_hash'] for r in data[a]}&{r['sentence_hash'] for r in data[b]}
model=SenseModel.load();selection=json.loads((run/'selection.json').read_text())
assert model.params==selection['params']
best=max(selection['development'],key=lambda r:r['accuracy'])
assert model.params==best['params']
assert sum(sum(c.values()) for c in model.counts.values())==len(data['train'])
comp=pickle.load(open(ROOT/'data/cache/semcor-components-v1-test.pkl','rb'))
pred=[model.rank_components(c)[0]['sense'] for c in comp]
base=[model.rank_components(c,{'nb':0.,'vector':0.})[0]['sense'] for c in comp]
hits=np.array([p==r['sense'] for p,r in zip(pred,data['test'])])
basehits=np.array([p==r['sense'] for p,r in zip(base,data['test'])])
metrics=json.loads((run/'metrics.json').read_text())
assert abs(hits.mean()-metrics['test']['accuracy'])<1e-12
group=defaultdict(list)
for i,r in enumerate(data['test']):group[r['document']].append(i)
counts=np.array([[sum(hits[idx]-basehits[idx].astype(int)),len(idx)] for idx in group.values()])
rng=np.random.default_rng(4701);draw=rng.integers(0,len(counts),(2000,len(counts)))
res=counts[draw].sum(axis=1);delta=res[:,0]/res[:,1]
interval=list(np.quantile(delta,[.025,.975]))
metrics['accuracy_gain_document_bootstrap_95ci']=interval
(run/'metrics.json').write_text(json.dumps(metrics,indent=2))
evaluation=json.loads((run/'evaluation.json').read_text())
basic=next(r for r in evaluation['tsar'] if r['name']=='basic')
old=json.loads((ROOT/'runs/initial-20260924/lexical.json').read_text())
oldbasic=next(r for r in old['tsar'] if r['name']=='learned')
assert abs(basic['top1']-oldbasic['top1_gold_match'])<1e-12
for r in evaluation['tsar']:
    rows=evaluation['outputs'][r['name']]
    assert r['correct']==sum(x['gold_match'] for x in rows)
    assert r['edits']==sum(x['changed'] for x in rows)
    assert all(x['output']==x['original'] for x in rows if not x['changed'])
assert len(evaluation['samples'])==20
print('Verified document and sentence separation, training counts, dev selection, test accuracy, unchanged baseline, and 20 samples.')
print('Accuracy gain bootstrap interval:',interval)
