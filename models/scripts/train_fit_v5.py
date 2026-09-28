"""Context block with more data: SWORDS dev + test (generated pairs) for the checker and fit stacker.

SWORDS test was only an exploratory report set; TSAR test now plays that role. Comparison: 5-fold grouped CV over
the combined pool, training on dev rows only versus dev + test rows, scored on the same held-out generated pairs.
"""
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score,log_loss
from cleartext.data import ROOT
from cleartext import ensemble as E
V5=ROOT/'runs/ensemble-v5-20260927';V5.mkdir(exist_ok=True)
from cleartext.ensemble import checker_model
d=pickle.load(open(ROOT/'data/cache/ensemble-v2-swords-dev.pkl','rb'));t=pickle.load(open(ROOT/'data/cache/ensemble-v2-swords-test.pkl','rb'))
k=t['keep']
X=np.r_[d['X'],t['X'][k]];y=np.r_[d['y'],t['y'][k]];g=np.r_[d['groups'],t['groups'][k]];gen=np.r_[d['generated'],t['generated'][k]]
rows=list(d['rows'])+[t['rows'][i] for i in np.flatnonzero(k)];is_test=np.r_[np.zeros(len(d['y']),bool),np.ones(k.sum(),bool)]
assert not set(d['groups'])&set(t['groups'][k])
names=json.loads((ROOT/'runs/ensemble-v2-20260927/fit_selection.json').read_text())['members'];CHECK=names.index('swords_checker')
C=json.loads((ROOT/'runs/ensemble-v2-20260927/config.json').read_text())['stacker_C']
folds=list(GroupKFold(n_splits=5).split(X,y,g))
res={}
for label,use in [('dev only',~is_test),('dev + test',np.ones(len(y),bool))]:
    p=np.zeros(len(y));Xc=X.copy()
    for a,b in folds:
        a=a[use[a]]
        # Inner cross-fit for the checker column on the training part, so the stacker never sees in-sample checker scores.
        inner=list(GroupKFold(n_splits=4).split(a,y[a],g[a]))
        for ia,ib in inner:
            Xc[a[ib],CHECK]=checker_model().fit([rows[i] for i in a[ia]],y[a[ia]]).predict_proba([rows[i] for i in a[ib]])[:,1]
        Xc[b,CHECK]=checker_model().fit([rows[i] for i in a],y[a]).predict_proba([rows[i] for i in b])[:,1]
        p[b]=E.StackedFit(None,C).fit(Xc[a],y[a]).proba(Xc[b])
    res[label]={'auc_generated':float(roc_auc_score(y[gen],p[gen])),'log_loss_generated':float(log_loss(y[gen],np.clip(p[gen],1e-6,1-1e-6)))}
    print(label,json.dumps(res[label]),flush=True)
# Final: cross-fitted checker column on all rows, then stacker and full checker on all rows.
Xf=X.copy()
for a,b in folds:Xf[b,CHECK]=checker_model().fit([rows[i] for i in a],y[a]).predict_proba([rows[i] for i in b])[:,1]
checker=checker_model().fit(rows,y);stacker=E.StackedFit(None,C).fit(Xf,y)
pickle.dump(checker,open(V5/'checker.pkl','wb'));pickle.dump(stacker,open(V5/'stacker.pkl','wb'))
json.dump({'comparison':res,'pairs':int(len(y)),'generated_pairs':int(gen.sum()),'contexts':int(len(set(g))),'weights':dict(zip(names,map(float,stacker.model.coef_[0])))},
          open(V5/'fit_training.json','w'),indent=2)
# Run config for build_candidate_tables.py (REFINE_RUN=runs/ensemble-v5-20260927): v2 settings with the new checker.
config=json.loads((ROOT/'runs/ensemble-v2-20260927/config.json').read_text());config['checker']=str((V5/'checker.pkl').relative_to(ROOT))
(V5/'config.json').write_text(json.dumps(config,indent=2))
print('SAVED',V5)
