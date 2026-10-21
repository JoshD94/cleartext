"""Round 10: target part of speech as decision features (v5 tables, round-7 protocol)."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import cleartext.refine as R
from cleartext.data import ROOT
# Part-of-speech columns are added explicitly below, so the library toggle is off here.
_,S=R.tables('v5');R.SPECIFICITY=True;R.POS_FEATURES=False;dev=S['benchls_dev']+S['benchls_holdout'];tsar=S['tsar_test']
def pos_cols(cases):
    return np.array([[float((case.get('target_info') or (None,'x'))[1]==p) for p in 'nvar'] for case in cases for c in case['candidates']])
out={}
for label,use in [('v5 features',False),('+ target part of speech',True)]:
    X,y,g,k=R.flat_specific(dev);Xt,_,_,kt=R.flat_specific(tsar)
    if use:X,Xt=np.c_[X,pos_cols(dev)],np.c_[Xt,pos_cols(tsar)]
    net,loss,t,nets=R.cv_decision(X,y,g,k,dev,R.AverageDecision)
    res=R.evaluate(tsar,R.accept_rule(dict(zip(kt,R.AverageDecision().fit(X,y).predict_proba(Xt)[:,1])),t))
    out[label]={'dev_net':net,'dev_nets':nets,'dev_log_loss':loss,'threshold':t,'tsar':res}
    print(f"{label:<26} dev net {net:6.1f} {nets} log-loss {loss:.4f} t={t} | tsar {res['correct']}/{res['edits']} net {res['net']}",flush=True)
json.dump(out,open(ROOT/'runs/ensemble-v5-20260927/refine_round10.json','w'),indent=2)
