import sys,json,gzip,pickle,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
from cleartext.data import ROOT,RAW
from cleartext.features import parse,target_token
from cleartext.lexical import synsets,POS
from cleartext.semantics import vector_features,sense_scores
run=ROOT/'runs/context-20260924';run.mkdir(exist_ok=True)
started=time.time()
def data(split):
    cache=ROOT/f'data/cache/swords-vectors-v2-{split}.pkl'
    if cache.exists():return pickle.load(open(cache,'rb'))
    base,y,groups,records,skip=pickle.load(open(ROOT/f'data/cache/swords-features-{split}.pkl','rb'))
    d=json.load(gzip.open(RAW/f'swords_{split}.json.gz','rt'))
    ids=list(d['contexts']);docs=parse([d['contexts'][i]['context'] for i in ids],f'swords-{split}');byid=dict(zip(ids,docs))
    features=[];sense_cache={}
    for i,(b,r) in enumerate(zip(base,records)):
        c=d['substitutes'][r['id']];t=d['targets'][c['target_id']]
        doc=byid[t['context_id']];tok=target_token(doc,t['target'],t['offset'])
        if c['target_id'] not in sense_cache:sense_cache[c['target_id']]=sense_scores(doc,tok)
        candidate={'word':c['substitute'],'senses':[s.name() for s in synsets(tok.lemma_.lower(),POS.get(tok.pos_,'n')) if c['substitute'].lower().replace(' ','_') in {x.lower() for x in s.lemma_names()}]}
        features.append(vector_features(doc,tok,candidate,b,sense_cache[c['target_id']]))
    result=(features,y,groups)
    pickle.dump(result,open(cache,'wb'))
    print('FEATURES',split,len(y),flush=True)
    return result
def metrics(y,p,t):
    pred=p>=t
    return dict(precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),auc=float(roc_auc_score(y,p)),accepted=int(pred.sum()),n=len(y))
X,y,g=data('dev')
tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=4701).split(X,y,g))
models={};results=[]
for name,est in [('logistic',LogisticRegression(C=1,max_iter=1500)),('boosting',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=10,random_state=4701))]:
    model=make_pipeline(DictVectorizer(sparse=False),StandardScaler(),est)
    model.fit([X[i] for i in tr],y[tr]);p=model.predict_proba([X[i] for i in va])[:,1]
    curve=[dict(threshold=float(t),**metrics(y[va],p,t)) for t in np.arange(.05,.96,.025)]
    # Require a nontrivial sample. If no threshold reaches the development precision goal, abstain.
    eligible=[r for r in curve if r['precision']>=.70 and r['accepted']>=30]
    operating=max(eligible,key=lambda r:r['recall']) if eligible else dict(threshold=1.01,**metrics(y[va],p,1.01))
    results.append(dict(name=name,operating=operating,at_half=metrics(y[va],p,.5),curve=curve))
    models[name]=model
selected=max(results,key=lambda r:(r['operating']['recall'],r['at_half']['auc']))
config={'selected':selected['name'],'threshold':selected['operating']['threshold'],'results':results,
        'scope':'Development precision target 70%, at least 30 accepted pairs; otherwise keep originals. Pair classification does not certify meaning.',
        'split':{'training_pairs':len(tr),'validation_pairs':len(va),'train_contexts':len(set(g[tr])),'validation_contexts':len(set(g[va]))},
        'vectors':'en_core_web_md 3.8.0 static vectors averaged over content words; not contextual token embeddings'}
(run/'selection.json').write_text(json.dumps(config,indent=2))
pickle.dump(models[selected['name']],open(run/'checker.pkl','wb'))
# Freeze selection above before looking at reused test data.
Xt,yt,gt=data('test');assert not set(g)&set(gt)
pt=models[selected['name']].predict_proba(Xt)[:,1]
config['test']=metrics(yt,pt,config['threshold'])
config['test_at_half']=metrics(yt,pt,.5)
old=pickle.load(open(ROOT/'runs/initial-20260924/fit_model.pkl','rb'))
oldx=pickle.load(open(ROOT/'data/cache/swords-features-test.pkl','rb'))[0]
config['baseline_at_same_threshold']=metrics(yt,old.predict_proba(oldx)[:,1],config['threshold'])
config['runtime_seconds']=time.time()-started
(run/'metrics.json').write_text(json.dumps(config,indent=2))
print('RESULT',json.dumps({k:v for k,v in config.items() if k!='results'}),flush=True)
