"""Round 5: does 'candidate already appears in the sentence' help? Chosen on BenchLS dev (10-fold x3, grouped)."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import cleartext.refine as R
from cleartext.data import ROOT
names,S=R.tables();dev=S['benchls_dev']
def repeated_cv(X,y,g,k,C=.03):
    nets=[]
    for seed in range(3):
        rng=np.random.default_rng(seed);ug=np.unique(g);perm=dict(zip(ug,rng.permutation(len(ug))%10))
        fold=np.array([perm[x] for x in g]);p=np.zeros(len(y))
        for f in range(10):
            tr,te=fold!=f,fold==f;p[te]=R.accept_model(C).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
        nets.append(R.best_threshold(dev,dict(zip(k,p)))[1])
    return nets
out={}
for flag in [False,True]:
    R.IN_SENTENCE=flag;X,y,g,k=R.flat_listwise(dev);nets=repeated_cv(X,y,g,k)
    m=R.accept_model(.03).fit(X,y);res={}
    for s in ['benchls_holdout','tsar_test']:
        Xs,_,_,ks=R.flat_listwise(S[s]);res[s]=R.evaluate(S[s],R.accept_rule(dict(zip(ks,m.predict_proba(Xs)[:,1])),.40))
    out[str(flag)]={'dev_nets':nets,**res}
    print(f"in_sentence={flag!s:<5} dev net {nets} mean {np.mean(nets):.1f} | holdout {res['benchls_holdout']['correct']}/{res['benchls_holdout']['edits']} net {res['benchls_holdout']['net']}"
          f" | tsar {res['tsar_test']['correct']}/{res['tsar_test']['edits']} net {res['tsar_test']['net']}")
json.dump(out,open(ROOT/'runs/ensemble-v3-20260927/refine_round5.json','w'),indent=2)
