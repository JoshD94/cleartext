"""Context block: tune the SWORDS checker's boosting settings. Grouped 5-fold CV over SWORDS dev + test (generated
pairs), scored by checker AUC on generated pairs. Rebuild downstream only if AUC improves by >= 0.005."""
import sys,json,pickle,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from cleartext.data import ROOT
d=pickle.load(open(ROOT/'data/cache/ensemble-v2-swords-dev.pkl','rb'));t=pickle.load(open(ROOT/'data/cache/ensemble-v2-swords-test.pkl','rb'));k=t['keep']
y=np.r_[d['y'],t['y'][k]];g=np.r_[d['groups'],t['groups'][k]];gen=np.r_[d['generated'],t['generated'][k]]
rows=list(d['rows'])+[t['rows'][i] for i in np.flatnonzero(k)]
V=DictVectorizer(sparse=False).fit(rows);M=V.transform(rows)
folds=list(GroupKFold(n_splits=5).split(M,y,g));out=[]
for leaves,l2,it,lr in [(15,10,150,.1)]+list(itertools.product([7,15,31],[1,10,30],[150,400],[.05,.1])):
    p=np.zeros(len(y))
    for a,b in folds:p[b]=HistGradientBoostingClassifier(max_iter=it,learning_rate=lr,max_leaf_nodes=leaves,l2_regularization=l2,random_state=4701).fit(M[a],y[a]).predict_proba(M[b])[:,1]
    r={'leaves':leaves,'l2':l2,'iter':it,'lr':lr,'auc_generated':float(roc_auc_score(y[gen],p[gen])),'auc_all':float(roc_auc_score(y,p))}
    out.append(r);print(json.dumps(r),flush=True)
cur=out[0];best=max(out,key=lambda r:r['auc_generated'])
print('CURRENT',cur);print('BEST',best,'gain',round(best['auc_generated']-cur['auc_generated'],4))
json.dump({'grid':out,'current':cur,'best':best},open(ROOT/'runs/ensemble-v5-20260927/refine_checker.json','w'),indent=2)
