"""Stage 2: train the stacked fit model on SWORDS dev pairs the v2 checker never saw, then freeze it.

Stage 3 (same script, after freezing): reused SWORDS test and TSAR test, plus sample outputs.
"""
import sys,json,gzip,pickle,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score,log_loss
from cleartext.data import ROOT,RAW,tsar_data
from cleartext.features import parse,target_token
from cleartext.lexical import synsets,POS
from cleartext import ensemble as E
from cleartext.ensemble_pipeline import EnsembleClearText,build_sense,fit_members,utility,RUN
started=time.time()
config=json.loads((RUN/'config.json').read_text())
sense=build_sense(pickle.load(open(RUN/'sense_model.pkl','rb')),config)
checker=pickle.load(open(ROOT/'runs/context-20260924/checker.pkl','rb'))
stacker=E.StackedFit(fit_members(sense,checker))
difficulty=E.RegressorDifficulty(pickle.load(open(ROOT/'runs/initial-20260924/word_model.pkl','rb')))
names=[m.name for m in stacker.members]

def pairs(split,keep=None):
    """Member scores for SWORDS pairs. keep: indices into the cached feature order, or None for all."""
    path=ROOT/f'data/cache/ensemble-swords-v1-{split}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    base,y,groups,records,skip=pickle.load(open(ROOT/f'data/cache/swords-features-{split}.pkl','rb'))
    d=json.load(gzip.open(RAW/f'swords_{split}.json.gz','rt'))
    ids=list(d['contexts']);docs=parse([d['contexts'][i]['context'] for i in ids],f'swords-{split}');byid=dict(zip(ids,docs))
    keep=range(len(records)) if keep is None else keep
    bytarget={}
    for i in keep:
        c=d['substitutes'][records[i]['id']];bytarget.setdefault(c['target_id'],[]).append(i)
    X=np.zeros((len(records),len(names)));done=np.zeros(len(records),bool);member=np.zeros(len(records),bool)
    target=np.array(['']*len(records),dtype=object);gain=np.zeros(len(records))
    for n,(tid,idx) in enumerate(bytarget.items()):
        t=d['targets'][tid];doc=byid[t['context_id']];tok=target_token(doc,t['target'],t['offset'])
        pos=POS.get(tok.pos_,'n');senses=synsets(tok.lemma_.lower(),pos)
        cands=[]
        for i in idx:
            sub=d['substitutes'][records[i]['id']]['substitute'];key=sub.lower().replace(' ','_')
            lemma=wn.morphy(key,pos) or key
            cands.append({'word':sub,'lemma':lemma,'senses':[s.name() for s in senses if {key,lemma}&{l.lower() for l in s.lemma_names()}]})
        X[idx]=stacker.features(E.Slot(doc,tok),cands);done[idx]=True
        member[idx]=[bool(c['senses']) for c in cands];target[idx]=tid
        dd=difficulty.score([tok.text]+[c['word'] for c in cands]);gain[idx]=dd[0]-dd[1:]
        if n%100==0:print('SWORDS',split,n,'of',len(bytarget),flush=True)
    out=(X,np.asarray(y,bool),np.asarray(groups),done,member,target,gain)
    pickle.dump(out,open(path,'wb'));return out

def binary(y,p,t):
    pred=p>=t
    return {'threshold':float(t),'accepted':int(pred.sum()),'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'n':int(len(y))}

# Same split as train_context_v2.py: the checker was fit on `tr`; only `va` gives it out-of-sample scores.
Xv2,yv2,gv2=pickle.load(open(ROOT/'data/cache/swords-vectors-v2-dev.pkl','rb'))
tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=4701).split(Xv2,yv2,gv2))
X,y,g,done,member,target,gain=pairs('dev',keep=va)
assert done[va].all() and not done[tr].any()
X,y,g,member,target,gain=X[va],y[va],g[va],member[va],target[va],gain[va]
folds=list(GroupKFold(n_splits=5).split(X,y,g))
selection=[]
for C in [.03,.1,.3,1.,3.]:
    oof=np.zeros(len(y))
    for a,b in folds:oof[b]=E.StackedFit(stacker.members,C).fit(X[a],y[a]).proba(X[b])
    selection.append({'C':C,'oof':oof,'auc_wordnet':float(roc_auc_score(y[member],oof[member])),'log_loss':float(log_loss(y,oof))})
chosen=max(selection,key=lambda r:r['auc_wordnet'])
oof=chosen['oof']
def simulate(p,y,target,gain,member,cost):
    """Per target, apply the highest-utility generated candidate if its utility is positive (as the pipeline does)."""
    best={}
    for i in np.flatnonzero(member&(gain>0)):
        u=utility(p[i],gain[i],cost)
        if u>0 and (target[i] not in best or u>best[target[i]][0]):best[target[i]]=(u,i)
    correct=sum(bool(y[i]) for u,i in best.values())
    return {'error_cost':float(cost),'targets':int(len(set(target[member]))),'edits':len(best),'correct':correct,'wrong':len(best)-correct}
curve=[simulate(oof,y,target,gain,member,c) for c in [0.,.01,.02,.03,.05,.075,.1,.15,.2,.3,.5]]
# An edit should be right more often than wrong: maximize correct minus wrong edits, then prefer more edits.
operating=max(curve,key=lambda r:(r['correct']-r['wrong'],r['edits']))
stacker.C=chosen['C'];stacker.fit(X,y)
config.update(error_cost=operating['error_cost'],difficulty_cutoff=.3,stacker_C=chosen['C'])
members=stacker.members;stacker.members=None
pickle.dump(stacker,open(RUN/'stacker.pkl','wb'));stacker.members=members
(RUN/'config.json').write_text(json.dumps(config,indent=2))
fit_selection={'members':names,'weights':stacker.weights(),'C_search':[{k:v for k,v in r.items() if k!='oof'} for r in selection],
               'operating_point':operating,'curve':curve,'training_pairs':int(len(y)),'wordnet_pairs':int(member.sum()),
               'training_contexts':int(len(set(g))),'criterion':'5-fold grouped CV on SWORDS dev contexts held out from the v2 checker; C by WordNet-pair AUC; error_cost by out-of-fold per-target correct minus wrong edits'}
(RUN/'fit_selection.json').write_text(json.dumps(fit_selection,indent=2))
print('FROZEN',json.dumps({k:fit_selection[k] for k in ['weights','operating_point','training_pairs','wordnet_pairs']}),flush=True)

# ---- Frozen. Everything below reuses previously seen test sets and is exploratory. ----
base,yt_all,gt_all,records,_=pickle.load(open(ROOT/'data/cache/swords-features-test.pkl','rb'))
Xv2t=pickle.load(open(ROOT/'data/cache/swords-vectors-v2-test.pkl','rb'))[0]
wn_idx=[i for i,x in enumerate(Xv2t) if x['wordnet_member']]
Xt,yt,gt,done_t,member_t,target_t,gain_t=pairs('test',keep=wn_idx)
p=np.zeros(len(yt));m=done_t&member_t;p[m]=stacker.proba(Xt[m])
swords={'stacked':{'auc':float(roc_auc_score(yt[m],p[m])),**simulate(p,yt,target_t,gain_t,m,config['error_cost'])},
        'stacked_all_costs':[simulate(p,yt,target_t,gain_t,m,c) for c in [0.,.02,.05,.1,.2,.3]],
        'members_auc':{n:float(roc_auc_score(yt[m],Xt[m][:,j])) for j,n in enumerate(names) if np.std(Xt[m][:,j])>0}}
print('SWORDS TEST',json.dumps(swords),flush=True)

pipeline=EnsembleClearText.load()
rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test')
outputs=[];lines=[]
for row,doc in zip(rows,docs):
    tok=target_token(doc,row['target']);rank=pipeline.rank_word(doc,tok)
    sel=rank['selected'];answer=sel['word'] if sel else row['target']
    outputs.append({'id':row['id'],'original':row['text'],'target':row['target'],'replacement':answer,'output':sel['output'] if sel else row['text'],
                    'changed':sel is not None,'gold_match':bool(sel and answer.lower() in row['gold']),'gold':row['gold'][:5],
                    'senses':rank['senses'],'candidates':[{k:c[k] for k in ['word','gain','fit','utility','rejections']} for c in rank['candidates']]})
    lines.append(row['text']+'\t'+row['target']+'\t'+answer.lower())
path=RUN/'tsar_ensemble.tsv';path.write_text('\n'.join(lines)+'\n')
subprocess.run([sys.executable,str(RAW/'tsar_eval.py'),'--gold_file',str(RAW/'tsar_test.tsv'),'--predictions_file',str(path),'--output_file',str(RUN/'tsar_ensemble_official.txt')],check=True,capture_output=True)
def summary(out):
    e=sum(r['changed'] for r in out);c=sum(r['gold_match'] for r in out)
    return {'n':len(out),'edits':e,'correct':c,'match_among_edits':c/e if e else 0.,'wrong_edits':e-c}
old=json.loads((ROOT/'runs/semcor-20260924/evaluation.json').read_text())['outputs']
tsar={'dictionary_basic':summary(old['basic']),'previous_sense_and_guards':summary(old['sense_and_guards']),'ensemble':summary(outputs)}
# Diagnostic sweep only; error_cost above was frozen from SWORDS dev. Guardrails are not rerun here.
sweep=[]
for cost in [0.,.02,.05,.1,.2,.3]:
    e=c=0
    for row,r in zip(rows,outputs):
        ok=[x for x in r['candidates'] if x['gain']>0 and utility(x['fit'],x['gain'],cost)>0]
        if ok:
            best=max(ok,key=lambda x:utility(x['fit'],x['gain'],cost));e+=1;c+=best['word'].lower() in row['gold']
    sweep.append({'error_cost':cost,'edits':e,'correct':c})
tsar['diagnostic_sweep']=sweep
print('TSAR',json.dumps({k:v for k,v in tsar.items() if k!='diagnostic_sweep'}),flush=True)

examples=[r['original'] for r in json.loads((ROOT/'runs/initial-20260924/basic_examples_20.json').read_text())]
examples+=['The medication alleviates pain but can cause drowsiness.','Heavy rainfall may exacerbate the flooding.',
           'The contract stipulates that payment is due monthly.','She deposited the money at the bank.']
prev={r['original']:r['output'] for r in json.loads((ROOT/'runs/semcor-20260924/evaluation.json').read_text())['samples']}
samples=[{'previous':prev.get(s),**pipeline.analyze(s,structure=False,phrases=False)} for s in examples]
for s in samples:print('SAMPLE',s['original'],'=>',s['output'],'| previous:',s['previous'],flush=True)
(RUN/'evaluation.json').write_text(json.dumps({'swords_test':swords,'tsar':tsar,'tsar_outputs':outputs,'samples':samples,
    'note':'SWORDS test and TSAR test were used in earlier experiments; these are exploratory, not fresh final tests. No labels from them fit or selected any model here.',
    'runtime_seconds':time.time()-started},indent=2,default=str))
print('COMPLETE',flush=True)
