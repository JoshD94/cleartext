"""Round 4: swap the decision model (logistic, gradient boosting, their average) on round-3 features."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from cleartext.data import ROOT
from cleartext.refine import accept_model,accept_rule,best_threshold,evaluate,flat_listwise,oof_probs,tables
names,S=tables();dev=S['benchls_dev']
def boosting(C):
    # C is reused as the leaf-count knob so the same search loop applies.
    return HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=int(C),l2_regularization=1.,min_samples_leaf=20,random_state=4701)
class Average:
    """Mean probability of the logistic and boosting members; itself a drop-in classifier."""
    def __init__(self,C):self.members=[accept_model(.03),boosting(C)]
    def fit(self,X,y):
        for m in self.members:m.fit(X,y)
        return self
    def predict_proba(self,X):return np.mean([m.predict_proba(X) for m in self.members],axis=0)
X,y,g,k=flat_listwise(dev)
rows={s:flat_listwise(S[s]) for s in ['benchls_holdout','tsar_test']}
out={}
for label,model,grid in [('logistic',accept_model,[.03,.1,.3]),('gradient boosting',boosting,[4,8,16]),('average of both',Average,[4,8,16])]:
    best=None
    for C in grid:
        P=dict(zip(k,oof_probs(X,y,g,C,model=model)));t,net=best_threshold(dev,P)
        if best is None or net>best[2]:best=(C,t,net,P)
    C,t,net,P=best;m=model(C).fit(X,y);probs={'benchls_dev':P}
    for s,(Xs,_,_,ks) in rows.items():probs[s]=dict(zip(ks,m.predict_proba(Xs)[:,1]))
    out[label]={'C':C,'threshold':t,**{s:evaluate(S[s],accept_rule(probs[s],t)) for s in S}}
    r=out[label];print(f"{label:<18} C={C} t={t}  dev net {r['benchls_dev']['net']} ({r['benchls_dev']['correct']}/{r['benchls_dev']['edits']})"
                       f" | holdout {r['benchls_holdout']['correct']}/{r['benchls_holdout']['edits']} net {r['benchls_holdout']['net']}"
                       f" | tsar {r['tsar_test']['correct']}/{r['tsar_test']['edits']} net {r['tsar_test']['net']}",flush=True)
json.dump(out,open(ROOT/'runs/ensemble-v2-20260927/refine_round4.json','w'),indent=2)
