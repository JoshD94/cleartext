"""Round 8 (antonym flag dropped: generated candidates are never WordNet antonyms). Signals from existing blocks on top of v4 features. Dev = all BenchLS (10-fold x3 grouped); TSAR report-only."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import cleartext.refine as R
from cleartext.data import ROOT
names,S=R.tables();R.SPECIFICITY=True
dev=S['benchls_dev']+S['benchls_holdout'];tsar=S['tsar_test'];extras=R.load_extras()
base={s:R.flat_specific(c) for s,c in [('dev',dev),('tsar',tsar)]}
def with_groups(split,cases,groups):
    X,y,g,k=base[split]
    if not groups:return X,y,g,k
    add=np.array([R.extra_values(extras,case,c,groups) for case in cases for c in case['candidates']])
    return np.c_[X,add],y,g,k
def antonym_rule(prob,t):
    def choose(case):
        ok=[c for c in case['candidates'] if c['guard'] and prob[case['id'],c['word']]>=t and not R.extra_values(extras,case,c,['antonym'])[0]]
        return max(ok,key=lambda c:prob[case['id'],c['word']]) if ok else None
    return choose
out={}
for label,groups in [('v4 features',[]),('+ detector',['detector']),('+ sentence level',['sentence']),('+ both',['detector','sentence'])]:
    X,y,g,k=with_groups('dev',dev,groups);Xt,_,_,kt=with_groups('tsar',tsar,groups)
    net,loss,t,nets=R.cv_decision(X,y,g,k,dev,R.AverageDecision)
    m=R.AverageDecision().fit(X,y);P=dict(zip(kt,m.predict_proba(Xt)[:,1]));res=R.evaluate(tsar,R.accept_rule(P,t))
    out[label]={'dev_net':net,'dev_nets':nets,'dev_log_loss':loss,'threshold':t,'tsar':res}
    print(f"{label:<20} dev net {net:6.1f} {nets} log-loss {loss:.4f} t={t} | tsar {res['correct']}/{res['edits']} net {res['net']}",flush=True)
json.dump(out,open(ROOT/'runs/ensemble-v4-20260927/refine_round8.json','w'),indent=2)
