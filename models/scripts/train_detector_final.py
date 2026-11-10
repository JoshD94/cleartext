"""Detector refinement: tune boosting settings on CWI dev, then refit on train+dev. CWI test read once at the end."""
import sys,json,pickle,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from cleartext.data import ROOT
from cleartext import detection as D
load=lambda p:pickle.load(open(p,'rb'))
word_model=load(ROOT/'runs/initial-20260924/word_model.pkl');context_model=load(ROOT/'runs/initial-20260924/context_model.pkl')
train,dev,test=(D.cwi_rows(s,word_model,context_model) for s in ['Train','Dev','Test'])
F={k:D.cwi_features(d) for k,d in [('train',train),('dev',dev),('test',test)]}
y={k:np.array([r['label'] for r in d]) for k,d in [('train',train),('dev',dev),('test',test)]}
grid=np.round(np.arange(.2,.6,.01),2)
def make(lr,leaves,it,l2):
    return make_pipeline(DictVectorizer(sparse=False),HistGradientBoostingClassifier(max_iter=it,learning_rate=lr,max_leaf_nodes=leaves,l2_regularization=l2,random_state=4701))
results=[]
for lr,leaves,it,l2 in itertools.product([.03,.05,.1],[15,31,63],[300,600],[0.,1.]):
    m=make(lr,leaves,it,l2).fit(F['train'],y['train']);p=m.predict_proba(F['dev'])[:,1]
    t=max(grid,key=lambda t:f1_score(y['dev'],p>=t));f=f1_score(y['dev'],p>=t)
    results.append({'lr':lr,'leaves':leaves,'iter':it,'l2':l2,'threshold':float(t),'dev_f1':float(f)})
results.sort(key=lambda r:-r['dev_f1']);best=results[0]
print('previous (lr .05, 31 leaves, 300 iter, l2 0): dev F1 0.778')
print('best on dev',json.dumps(best))
old=pickle.load(open(ROOT/'runs/ensemble-v3-20260927/detector.pkl','rb'))
final=make(best['lr'],best['leaves'],best['iter'],best['l2']).fit(F['train']+F['dev'],np.r_[y['train'],y['dev']])
tuned=make(best['lr'],best['leaves'],best['iter'],best['l2']).fit(F['train'],y['train'])
report={'old_detector_test_f1':float(f1_score(y['test'],old['model'].predict_proba(F['test'])[:,1]>=old['threshold'])),
        'tuned_train_only_test_f1':float(f1_score(y['test'],tuned.predict_proba(F['test'])[:,1]>=best['threshold'])),
        'tuned_train_dev_test_f1':float(f1_score(y['test'],final.predict_proba(F['test'])[:,1]>=best['threshold']))}
print('CWI TEST',json.dumps(report))
V5=ROOT/'runs/ensemble-v5-20260927'
pickle.dump({'model':final,'threshold':best['threshold'],'name':'gradient boosting, tuned, train+dev'},open(V5/'detector.pkl','wb'))
json.dump({'grid':results,'selected':best,'test':report},open(V5/'refine_detector2.json','w'),indent=2)
