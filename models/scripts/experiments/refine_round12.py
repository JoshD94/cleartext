"""Round 12: tune the decision model's own settings (logistic C, boosting leaves, mixing weight). v5 tables, round-7
protocol. Adopt only if the gain beats noise: mean net +5 or log-loss -0.001 versus the current setting."""
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import cleartext.refine as R
from cleartext.data import ROOT
_,S=R.tables('v5');dev=S['benchls_dev']+S['benchls_holdout'];tsar=S['tsar_test']
X,y,g,k=R.flat_specific(dev);Xt,_,_,kt=R.flat_specific(tsar);out=[]
for C,leaves,w in itertools.product([.01,.03,.1],[4,8],[.3,.5,.7]):
    make=lambda:R.WeightedDecision(C,leaves,w)
    net,loss,t,nets=R.cv_decision(X,y,g,k,dev,make)
    res=R.evaluate(tsar,R.accept_rule(dict(zip(kt,make().fit(X,y).predict_proba(Xt)[:,1])),t))
    out.append({'C':C,'leaves':leaves,'weight':w,'dev_net':net,'dev_nets':nets,'dev_log_loss':loss,'threshold':t,'tsar':res})
    print(f"C={C:<5} leaves={leaves} w={w}: dev net {net:6.1f} log-loss {loss:.4f} t={t} | tsar {res['correct']}/{res['edits']} net {res['net']}",flush=True)
json.dump(out,open(ROOT/'runs/ensemble-v5-20260927/refine_round12.json','w'),indent=2)
