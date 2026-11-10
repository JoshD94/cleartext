"""Round 3: listwise features (standing among the case's candidates), shared-prefix and spelling-variant flags."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cleartext.data import ROOT
from cleartext.refine import accept_model,accept_rule,best_threshold,evaluate,flat,flat_listwise,oof_probs,tables
names,S=tables();dev=S['benchls_dev']
out={}
for label,flatten in [('round 2 features',flat),('+ listwise, prefix, spelling',flat_listwise)]:
    X,y,g,k=flatten(dev)
    best=None
    for C in [.03,.1,.3,1.]:
        P=dict(zip(k,oof_probs(X,y,g,C)));t,net=best_threshold(dev,P)
        if best is None or net>best[2]:best=(C,t,net,P)
    C,t,net,P=best
    m=accept_model(C).fit(X,y)
    probs={'benchls_dev':P}
    for s in ['benchls_holdout','tsar_test']:
        Xs,_,_,ks=flatten(S[s]);probs[s]=dict(zip(ks,m.predict_proba(Xs)[:,1]))
    out[label]={'C':C,'threshold':t,**{s:evaluate(S[s],accept_rule(probs[s],t)) for s in S}}
    r=out[label];print(f"{label:<32} C={C} t={t}  dev net {r['benchls_dev']['net']} ({r['benchls_dev']['correct']}/{r['benchls_dev']['edits']})"
                       f" | holdout {r['benchls_holdout']['correct']}/{r['benchls_holdout']['edits']} net {r['benchls_holdout']['net']}"
                       f" | tsar {r['tsar_test']['correct']}/{r['tsar_test']['edits']} net {r['tsar_test']['net']}",flush=True)
json.dump(out,open(ROOT/'runs/ensemble-v2-20260927/refine_round3.json','w'),indent=2)
