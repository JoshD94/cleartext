"""Meaning block with more data: refit SemCor counts on train + dev (settings already chosen on dev), then carry the
change through the chain: SWORDS sense columns -> checker + stacker -> run config. Tables and the decision comparison
follow in refine_round11.py. SemCor test stays report-only.
"""
import sys,json,gzip,pickle,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from nltk.corpus import wordnet as wn
from cleartext.data import ROOT,RAW
from cleartext.features import parse,target_token
from cleartext.generation import resolve_target,relation_map
from cleartext.wsd import SenseModel
from cleartext import ensemble as E
from cleartext.ensemble_pipeline import build_sense
V5=ROOT/'runs/ensemble-v5-20260927';V6=ROOT/'runs/ensemble-v6-20260927';V6.mkdir(exist_ok=True)
started=time.time()
config=json.loads((V5/'config.json').read_text())
data,_=pickle.load(open(ROOT/'data/cache/semcor-records-v1.pkl','rb'))
model=SenseModel().fit(data['train']+data['dev']);model.params=config['semcor_params']
pickle.dump(model,open(V6/'sense_model.pkl','wb'))
old=pickle.load(open(ROOT/config['sense_model'],'rb'))
def accuracy(m):
    return float(np.mean([m.predict(r['words'],r['index'],r['lemma'],r['pos'])['sense']==r['sense'] for r in data['test']]))
semcor={'train_only_test_accuracy':accuracy(old),'train_dev_test_accuracy':accuracy(model)}
print('SEMCOR TEST',json.dumps(semcor),flush=True)
sense=build_sense(model,config)
names=json.loads((ROOT/'runs/ensemble-v2-20260927/fit_selection.json').read_text())['members']
members={'sense_fit':E.SenseFit(sense),'round_trip':E.RoundTripFit(sense),'candidate_sense_rank':E.CandidateSenseRank(sense)}
cols=[names.index(n) for n in members]

def sense_columns(split):
    """Recompute only the sense-dependent member columns; every other column is unchanged from the v2 cache."""
    path=ROOT/f'data/cache/ensemble-v6-swords-{split}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    c=pickle.load(open(ROOT/f'data/cache/ensemble-v2-swords-{split}.pkl','rb'))
    _,_,_,records,_=pickle.load(open(ROOT/f'data/cache/swords-features-{split}.pkl','rb'))
    d=json.load(gzip.open(RAW/f'swords_{split}.json.gz','rt'))
    ids=list(d['contexts']);docs=parse([d['contexts'][i]['context'] for i in ids],f'swords-{split}');byid=dict(zip(ids,docs))
    bytarget={}
    for i in np.flatnonzero(c['keep']):bytarget.setdefault(d['substitutes'][records[i]['id']]['target_id'],[]).append(i)
    X=c['X'].copy()
    for k,(tid,idx) in enumerate(bytarget.items()):
        t=d['targets'][tid];doc=byid[t['context_id']];tok=target_token(doc,t['target'],t['offset']);tg=resolve_target(tok)
        rel={a.lower():b for a,b in relation_map(tg[0],tg[1]).items()} if tg else {}
        cands=[]
        for i in idx:
            sub=d['substitutes'][records[i]['id']]['substitute'];key=sub.lower().replace(' ','_')
            lemma=(wn.morphy(key,tg[1]) if tg else None) or key;hit=rel.get(key) or rel.get(lemma.lower())
            cands.append({'word':sub,'lemma':lemma.replace('_',' '),'senses':list(hit[1]) if hit else [],'source':hit[0] if hit else None})
        slot=E.Slot(doc,tok,target=tg)
        for j,m in zip(cols,members.values()):X[idx,j]=m.score(slot,cands)
        if k%200==0:print('SWORDS',split,k,'of',len(bytarget),round(time.time()-started),'s',flush=True)
    out={**c,'X':X};pickle.dump(out,open(path,'wb'));return out
d=sense_columns('dev');t=sense_columns('test')
# Checker inputs do not depend on the sense model; reuse the v5 checker. Stacker refit on the new columns.
from cleartext.ensemble import checker_model
k=t['keep'];X=np.r_[d['X'],t['X'][k]];y=np.r_[d['y'],t['y'][k]];g=np.r_[d['groups'],t['groups'][k]]
rows=list(d['rows'])+[t['rows'][i] for i in np.flatnonzero(k)]
from sklearn.model_selection import GroupKFold
CHECK=names.index('swords_checker');C=json.loads((ROOT/'runs/ensemble-v2-20260927/config.json').read_text())['stacker_C']
for a,b in GroupKFold(n_splits=5).split(X,y,g):X[b,CHECK]=checker_model().fit([rows[i] for i in a],y[a]).predict_proba([rows[i] for i in b])[:,1]
pickle.dump(E.StackedFit(None,C).fit(X,y),open(V6/'stacker.pkl','wb'))
for f in ['checker.pkl','detector.pkl','sentence_model.pkl']:shutil.copy(V5/f,V6/f)
config.update(sense_model=str((V6/'sense_model.pkl').relative_to(ROOT)),checker=str((V6/'checker.pkl').relative_to(ROOT)),
              detector=str((V6/'detector.pkl').relative_to(ROOT)),sentence_model=str((V6/'sentence_model.pkl').relative_to(ROOT)))
(V6/'config.json').write_text(json.dumps(config,indent=2))
json.dump({'semcor':semcor,'runtime_seconds':time.time()-started},open(V6/'meaning_training.json','w'),indent=2)
print('COMPLETE',flush=True)
