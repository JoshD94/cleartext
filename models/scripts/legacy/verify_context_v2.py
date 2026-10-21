import json,pickle,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from sklearn.metrics import roc_auc_score
from cleartext.data import ROOT,RAW
run=ROOT/'runs/context-20260924'
m=json.loads((run/'metrics.json').read_text())
selection=json.loads((run/'selection.json').read_text())
assert m['selected']==selection['selected'] and m['threshold']==selection['threshold']
X,y,g=pickle.load(open(ROOT/'data/cache/swords-vectors-v2-test.pkl','rb'))
_,_,devg=pickle.load(open(ROOT/'data/cache/swords-vectors-v2-dev.pkl','rb'))
assert not set(g)&set(devg)
clf=pickle.load(open(run/'checker.pkl','rb'));p=clf.predict_proba(X)[:,1]
assert abs(roc_auc_score(y,p)-m['test']['auc'])<1e-12
assert int((p>=selection['threshold']).sum())==m['test']['accepted']
pm=json.loads((run/'phrase_metrics.json').read_text())
for r in pm['manifest']:
    split={'dev':'trial'}.get(r['split'],r['split'])
    assert hashlib.sha256((RAW/f'complex_multi_{split}.tsv').read_bytes()).hexdigest()==r['sha256']
review=json.loads((run/'review.json').read_text())
assert len(review['examples'])==20 and len(review['previous20'])==20
assert all(not any(e['stage']=='word' for e in r['edits']) for r in review['examples'])
print('Saved model metrics, separate contexts, data checksums and 20 examples verified.')
