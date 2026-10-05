"""Sentence difficulty block: add detector summaries (flagged count/fraction, mean/max hardness) to the CEFR features.

Same data and splits as the initial model (CEFR-SP, overlap removed). Model family chosen on dev MAE; test read once.
"""
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from cleartext.data import ROOT,sentence_data
from cleartext.features import parse,sentence_features,word_features
from cleartext import detection as D
load=lambda p:pickle.load(open(p,'rb'))
word_model=load(ROOT/'runs/initial-20260924/word_model.pkl');context_model=load(ROOT/'runs/initial-20260924/context_model.pkl')
detector=load(ROOT/'runs/ensemble-v5-20260927/detector.pkl');old=load(ROOT/'runs/initial-20260924/sentence_model.pkl')
from cleartext.detection import CONTENT
def detector_summary(docs):
    """Per sentence: how many content words the detector flags, and their mean and max hardness."""
    out=[]
    for doc in docs:
        words=[t for t in doc if t.is_alpha and t.pos_ in CONTENT and not t.is_stop]
        v=D.probabilities(detector,D.rows_for(doc,words,word_model.predict([word_features(t.text) for t in words]) if words else [],context_model))
        th=detector['threshold']
        out.append({'det_flagged':float((v>=th).sum()),'det_fraction':float((v>=th).mean()) if len(v) else 0.,
                    'det_mean':float(v.mean()) if len(v) else 0.,'det_max':float(v.max()) if len(v) else 0.})
    return out
sentences,_=sentence_data();F={};y={}
for split,rows in sentences.items():
    docs=parse([r['text'] for r in rows],f'cefr-{split}')
    base=sentence_features(docs,word_model);det=detector_summary(docs)
    F[split]=(base,[{**a,**b} for a,b in zip(base,det)]);y[split]=np.array([r['label'] for r in rows])
    print('features',split,len(rows),flush=True)
def model(leaves,lr):return make_pipeline(DictVectorizer(sparse=False),HistGradientBoostingRegressor(max_iter=400,learning_rate=lr,max_leaf_nodes=leaves,l2_regularization=5,early_stopping=True,random_state=4701))
res={'initial model (dev)':float(mean_absolute_error(y['dev'],np.clip(old.predict(F['dev'][0]),1,6)))}
best=None
for label,idx in [('base features',0),('+ detector summaries',1)]:
    for leaves in [15,31]:
        for lr in [.03,.1]:
            m=model(leaves,lr).fit(F['train'][idx],y['train']);mae=float(mean_absolute_error(y['dev'],np.clip(m.predict(F['dev'][idx]),1,6)))
            res[f'{label} leaves={leaves} lr={lr} (dev)']=mae
            if best is None or mae<best[0]:best=(mae,label,idx,m)
mae,label,idx,m=best
res['selected']=label;res['selected_test']=float(mean_absolute_error(y['test'],np.clip(m.predict(F['test'][idx]),1,6)))
res['initial_test']=float(mean_absolute_error(y['test'],np.clip(old.predict(F['test'][0]),1,6)))
for k,v in res.items():print(f'{k:<48} {v}')
pickle.dump({'model':m,'features':'detector' if idx else 'base'},open(ROOT/'runs/ensemble-v5-20260927/sentence_model.pkl','wb'))
json.dump(res,open(ROOT/'runs/ensemble-v5-20260927/refine_sentence.json','w'),indent=2)
