"""Round 7: larger dev. All of BenchLS (929) as dev, 10-fold x3 grouped by target word; TSAR test report-only.

Re-decides the two contested changes (round 4 model swap, round 6 specificity) with two dev metrics:
net correct at the best threshold, and out-of-fold log-loss (threshold-free, less noisy).
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
import cleartext.refine as R
from cleartext.data import ROOT
names,S=R.tables();dev=S['benchls_dev']+S['benchls_holdout'];tsar=S['tsar_test']
def boosting(C=4):return HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=4,l2_regularization=1.,min_samples_leaf=20,random_state=4701)
class Average:
    def __init__(self,C=.03):self.members=[R.accept_model(.03),boosting()]
    def fit(self,X,y):
        for m in self.members:m.fit(X,y)
        return self
    def predict_proba(self,X):return np.mean([m.predict_proba(X) for m in self.members],axis=0)
def cv(X,y,g,k,model):
    nets,losses,ts=[],[],[]
    for seed in range(3):
        rng=np.random.default_rng(seed);ug=np.unique(g);perm=dict(zip(ug,rng.permutation(len(ug))%10))
        fold=np.array([perm[x] for x in g]);p=np.zeros(len(y))
        for f in range(10):
            tr,te=fold!=f,fold==f;p[te]=model().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
        t,net=R.best_threshold(dev,dict(zip(k,p)));nets.append(net);ts.append(t);losses.append(log_loss(y,np.clip(p,1e-6,1-1e-6)))
    return nets,losses,float(np.median(ts))
out={}
for spec in [False,True]:
    R.SPECIFICITY=spec;X,y,g,k=R.flat_specific(dev);Xt,_,_,kt=R.flat_specific(tsar)
    for label,model in [('logistic',lambda:R.accept_model(.03)),('boosting',boosting),('average',Average)]:
        nets,losses,t=cv(X,y,g,k,model);m=model().fit(X,y)
        res=R.evaluate(tsar,R.accept_rule(dict(zip(kt,m.predict_proba(Xt)[:,1])),t))
        key=f'{label}{" + specificity" if spec else ""}'
        out[key]={'dev_net':nets,'dev_net_mean':float(np.mean(nets)),'dev_log_loss':float(np.mean(losses)),'threshold':t,'tsar':res}
        print(f"{key:<26} dev net {np.mean(nets):6.1f} {nets}  log-loss {np.mean(losses):.4f}  t={t} | tsar {res['correct']}/{res['edits']} net {res['net']}",flush=True)
json.dump(out,open(ROOT/'runs/ensemble-v3-20260927/refine_round7.json','w'),indent=2)
