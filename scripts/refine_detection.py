"""Detection round: which words should the pipeline try to simplify?

CWI 2018 English (News, WikiNews, Wikipedia; 20 annotators per sentence). Single-word targets; label = shared-task
binary (complex if any annotator marked it). Selection on CWI dev, reported once on CWI test.
"""
import sys,csv,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from wordfreq import zipf_frequency
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from cleartext.data import ROOT,RAW
from cleartext.features import parse,target_token,word_features
from cleartext.ensemble_pipeline import CONTENT
load=lambda p:pickle.load(open(p,'rb'))
word_model=load(ROOT/'runs/initial-20260924/word_model.pkl');context_model=load(ROOT/'runs/initial-20260924/context_model.pkl')

def rows(split):
    path=ROOT/f'data/cache/cwi-rows-v1-{split}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    raw=[r for r in csv.reader(open(RAW/f'cwi_english_{split}.tsv'),delimiter='\t',quoting=csv.QUOTE_NONE) if ' ' not in r[4].strip()]
    sents=sorted({r[1] for r in raw});docs=dict(zip(sents,parse(sents,f'cwi-{split}')))
    out=[]
    for r in raw:
        doc=docs[r[1]];tok=target_token(doc,r[4],int(r[2]))
        if tok is None:continue
        wf=word_features(tok.text);cf=word_features(tok.text,doc,tok,context=True)
        out.append({'label':r[9]=='1','prob':float(r[10]),'word':tok.text,'pos':tok.pos_,'stop':tok.is_stop,'entity':bool(tok.ent_type_),
                    'content':tok.pos_ in CONTENT and not tok.is_stop and not tok.ent_type_,'wf':wf,'cf':cf})
    ws=word_model.predict([r['wf'] for r in out]);cs=context_model.predict([r['cf'] for r in out])
    for r,a,b in zip(out,ws,cs):r['word_score'],r['context_score']=float(a),float(b)
    pickle.dump(out,open(path,'wb'));return out

def metrics(y,p):
    return {'f1':round(float(f1_score(y,p)),4),'precision':round(float(precision_score(y,p)),4),'recall':round(float(recall_score(y,p)),4),'flagged':int(p.sum()),'n':len(y)}
train,dev,test=rows('Train'),rows('Dev'),rows('Test')
y={k:np.array([r['label'] for r in d]) for k,d in [('train',train),('dev',dev),('test',test)]}
content={k:np.array([r['content'] for r in d]) for k,d in [('train',train),('dev',dev),('test',test)]}
out={}
def report(label,pred_dev,pred_test):
    out[label]={'dev':metrics(y['dev'],pred_dev),'test':metrics(y['test'],pred_test)}
    print(f"{label:<52} dev F1 {out[label]['dev']['f1']:.3f} | test F1 {out[label]['test']['f1']:.3f} (P {out[label]['test']['precision']:.2f} R {out[label]['test']['recall']:.2f})",flush=True)
score=lambda d,key:np.array([r[key] for r in d])
grid=np.round(np.arange(.1,.6,.01),2)
report('current: word model >= 0.30, content words',(score(dev,'word_score')>=.3)&content['dev'],(score(test,'word_score')>=.3)&content['test'])
for key,label in [('word_score','word model'),('context_score','context model')]:
    t=max(grid,key=lambda t:f1_score(y['dev'],(score(dev,key)>=t)&content['dev']))
    report(f'{label}, threshold {t} tuned on CWI dev',(score(dev,key)>=t)&content['dev'],(score(test,key)>=t)&content['test'])
def feats(d):
    return [{**{k:v for k,v in r['cf'].items()},'word_score':r['word_score'],'context_score':r['context_score'],'stop':float(r['stop']),
             'entity':float(r['entity']),'lower':float(r['word'].islower())} for r in d]
F={k:feats(d) for k,d in [('train',train),('dev',dev),('test',test)]}
models={'logistic':lambda:make_pipeline(DictVectorizer(sparse=False),StandardScaler(),LogisticRegression(C=1,max_iter=3000)),
        'gradient boosting':lambda:make_pipeline(DictVectorizer(sparse=False),HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=31,random_state=4701))}
fitted={}
for name,make in models.items():
    m=make().fit(F['train'],y['train']);pd_,pt=m.predict_proba(F['dev'])[:,1],m.predict_proba(F['test'])[:,1]
    t=max(grid,key=lambda t:f1_score(y['dev'],pd_>=t))
    fitted[name]=(m,t);report(f'{name} on CWI train (CompLex scores + features), t={t}',pd_>=t,pt>=t)
best=max(fitted,key=lambda n:out[[k for k in out if k.startswith(n)][0]]['dev']['f1'])
m,t=fitted[best];pickle.dump({'model':m,'threshold':float(t),'name':best},open(ROOT/'runs/ensemble-v3-20260927/detector.pkl','wb'))
json.dump({'results':out,'selected':best,'threshold':float(t),'label':'CWI 2018 binary (any of 20 annotators)'},open(ROOT/'runs/ensemble-v3-20260927/refine_detection.json','w'),indent=2)
print('SELECTED on dev:',best,t)
