"""Sentence block as a safety check: reject word edits that raise the predicted reading level (v5 sentence model).

Level change cached per (case id, word) for BenchLS and TSAR candidates. Rule thresholds compared on BenchLS
(all 929, out-of-fold decision probabilities, v5 tables); TSAR report-only.
"""
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import cleartext.refine as R
from cleartext.data import ROOT,benchls_data,tsar_data
from cleartext.features import parse,target_token,nlp
from cleartext.ensemble_pipeline import EnsembleClearText,LATEST,substitute
pipe=EnsembleClearText.load(LATEST)
def levels(name,rows,docs,tokens,names):
    path=ROOT/f'data/cache/refine-level-v5-{name}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    out={};cases={c['id']:c for c in names}
    for k,(r,doc,tok) in enumerate(zip(rows,docs,tokens)):
        case=cases.get(r['id'])
        if tok is None or not case or not case['candidates']:continue
        base=pipe.reading_level(doc)
        for c,nd in zip(case['candidates'],nlp().pipe([substitute(doc,tok,c['word']) for c in case['candidates']])):
            out[r['id'],c['word']]=pipe.reading_level(nd)-base
        if k%100==0:print('LEVEL',name,k,flush=True)
    pickle.dump(out,open(path,'wb'));return out
_,S=R.tables('v5');b=benchls_data();delta={}
for split in ['dev','holdout']:
    rows=b[split];docs=parse([r['text'] for r in rows],f'benchls-{split}')
    tok=[target_token(d,r['target'],sum(len(w)+1 for w in r['text'].split(' ')[:r['index']])) for r,d in zip(rows,docs)]
    delta.update(levels(f'benchls-{split}',rows,docs,tok,S[f'benchls_{split}']))
rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test')
delta.update(levels('tsar-test',rows,docs,[target_token(d,r['target']) for r,d in zip(rows,docs)],S['tsar_test']))
R.SPECIFICITY=True;dev=S['benchls_dev']+S['benchls_holdout'];tsar=S['tsar_test']
X,y,g,k=R.flat_specific(dev);Xt,_,_,kt=R.flat_specific(tsar)
oof=[]
for seed in range(3):
    rng=np.random.default_rng(seed);ug=np.unique(g);perm=dict(zip(ug,rng.permutation(len(ug))%10));fold=np.array([perm[x] for x in g]);p=np.zeros(len(y))
    for f in range(10):p[fold==f]=R.AverageDecision().fit(X[fold!=f],y[fold!=f]).predict_proba(X[fold==f])[:,1]
    oof.append(dict(zip(k,p)))
Pt=dict(zip(kt,R.AverageDecision().fit(X,y).predict_proba(Xt)[:,1]))
def rule(P,t,tau):
    def choose(case):
        ok=[c for c in case['candidates'] if c['guard'] and P[case['id'],c['word']]>=t and delta.get((case['id'],c['word']),0.)<=tau]
        return max(ok,key=lambda c:P[case['id'],c['word']]) if ok else None
    return choose
out={}
for tau in [np.inf,.2,.1,.05,0.]:
    nets=[R.evaluate(dev,rule(P,.45,tau))['net'] for P in oof];r=R.evaluate(tsar,rule(Pt,.45,tau))
    out[str(tau)]={'dev_nets':nets,'tsar':r}
    print(f"reject if level rises by > {tau}: dev net {np.mean(nets):6.1f} {nets} | tsar {r['correct']}/{r['edits']} net {r['net']}",flush=True)
json.dump(out,open(LATEST/'refine_level_guard.json','w'),indent=2)
