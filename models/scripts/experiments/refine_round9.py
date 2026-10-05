"""Round 9: decision model on tables built with the v5 context block (SWORDS dev+test). Same protocol as round 7."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import cleartext.refine as R
from cleartext.data import ROOT
R.SPECIFICITY=True;out={}
for version in ['v2','v5']:
    names,S=R.tables(version);dev=S['benchls_dev']+S['benchls_holdout'];tsar=S['tsar_test']
    X,y,g,k=R.flat_specific(dev);Xt,_,_,kt=R.flat_specific(tsar)
    net,loss,t,nets=R.cv_decision(X,y,g,k,dev,R.AverageDecision)
    m=R.AverageDecision().fit(X,y);res=R.evaluate(tsar,R.accept_rule(dict(zip(kt,m.predict_proba(Xt)[:,1])),t))
    out[version]={'dev_net':net,'dev_nets':nets,'dev_log_loss':loss,'threshold':t,'tsar':res}
    print(f"fit from {version}: dev net {net:6.1f} {nets} log-loss {loss:.4f} t={t} | tsar {res['correct']}/{res['edits']} net {res['net']}",flush=True)
json.dump(out,open(ROOT/'runs/ensemble-v5-20260927/refine_round9.json','w'),indent=2)
