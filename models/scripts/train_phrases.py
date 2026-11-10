import sys,json,csv,pickle,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from cleartext.data import ROOT,RAW,fetch,word_data
from cleartext.phrase_complexity import phrase_features
run=ROOT/'runs/context-20260924';run.mkdir(exist_ok=True)
commit=json.loads((RAW/'CompLex.commit.json').read_text())['sha']
splits=word_data();manifest=[]
for split,d,s in [('train','train','train'),('dev','trial','trial'),('test','test-labels','test')]:
    url=f'https://raw.githubusercontent.com/MMU-TDMLab/CompLex/{commit}/{d}/lcp_multi_{s}.tsv'
    path=fetch(url,RAW/f'complex_multi_{s}.tsv')
    rows=list(csv.DictReader(path.open(),delimiter='\t',quoting=csv.QUOTE_NONE))
    for r in rows:r['label']=float(r['complexity'])
    splits[split]=[{**r,'multi':False} for r in splits[split]]+[{**r,'multi':True} for r in rows]
    manifest.append({'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'split':split,'multi_count':len(rows)})
X={s:[phrase_features(r['token']) for r in rows] for s,rows in splits.items()}
y={s:[r['label'] for r in rows] for s,rows in splits.items()}
models={};results=[]
for name,est in [('ridge',Ridge(alpha=10)),('boosting',HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=5,random_state=4701))]:
    model=make_pipeline(DictVectorizer(sparse=False),StandardScaler(),est)
    model.fit(X['train'],y['train']);models[name]=model
    pred=np.clip(model.predict(X['dev']),0,1)
    results.append({'name':name,'dev_mae':float(mean_absolute_error(y['dev'],pred))})
chosen=min(results,key=lambda r:r['dev_mae'])['name']
(run/'phrase_selection.json').write_text(json.dumps({'selected':chosen,'results':results},indent=2))
pickle.dump(models[chosen],open(run/'phrase_model.pkl','wb'))
pred=np.clip(models[chosen].predict(X['test']),0,1)
multi=np.array([r['multi'] for r in splits['test']])
out={'selected':chosen,'results':results,'test_mae':float(mean_absolute_error(y['test'],pred)),
     'multiword_test_mae':float(mean_absolute_error(np.array(y['test'])[multi],pred[multi])),
     'multiword_test_n':int(multi.sum()),'manifest':manifest,
     'counts':{s:len(v) for s,v in y.items()},
     'limitation':'Mostly two-word phrase targets; longer expression scores are extrapolations. Candidate inventory is authored, not learned.'}
(run/'phrase_metrics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out),flush=True)
