"""Ensemble v2: expanded WordNet generation, POS fallback, and a cross-fitted checker over all SWORDS dev.

Reuses the stage-1 sense ensemble from runs/ensemble-20260927 unchanged. Selection uses SWORDS dev only;
SWORDS test and TSAR test below are reused, exploratory checks.
"""
import sys,json,gzip,pickle,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from cleartext.data import ROOT,RAW,tsar_data
from cleartext.features import parse,target_token
from cleartext.generation import resolve_target,relation_map
from cleartext import ensemble as E
from cleartext.ensemble import checker_model
from cleartext.ensemble_pipeline import EnsembleClearText,build_sense,fit_members,utility,RUN,RUN_V1
RUN.mkdir(exist_ok=True);started=time.time()
v1=json.loads((RUN_V1/'config.json').read_text())
config={k:v1[k] for k in ['semcor_params','semcor_temperature','gloss_temperature','sense_weights','sense_evidence']}
config.update(sense_model=str((RUN_V1/'sense_model.pkl').relative_to(ROOT)),checker=str((RUN/'checker.pkl').relative_to(ROOT)),
              generator='expanded',difficulty_cutoff=.3)
sense=build_sense(pickle.load(open(ROOT/config['sense_model'],'rb')),config)
class Placeholder:
    """Checker column is filled later with cross-fitted probabilities."""
    def predict_proba(self,rows):return np.zeros((len(rows),2))
members=fit_members(sense,Placeholder(),expanded=True);names=[m.name for m in members]
CHECK=names.index('swords_checker')
stacker=E.StackedFit(members)
difficulty=E.RegressorDifficulty(pickle.load(open(ROOT/'runs/initial-20260924/word_model.pkl','rb')))

def pairs(split):
    """Member scores, checker inputs and generation metadata for SWORDS pairs.

    dev: every pair (the checker and stacker train on all of them). test: only pairs our generator can propose.
    """
    path=ROOT/f'data/cache/ensemble-v2-swords-{split}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    _,y,groups,records,_=pickle.load(open(ROOT/f'data/cache/swords-features-{split}.pkl','rb'))
    d=json.load(gzip.open(RAW/f'swords_{split}.json.gz','rt'))
    ids=list(d['contexts']);docs=parse([d['contexts'][i]['context'] for i in ids],f'swords-{split}');byid=dict(zip(ids,docs))
    bytarget={}
    for i,r in enumerate(records):bytarget.setdefault(d['substitutes'][r['id']]['target_id'],[]).append(i)
    n=len(records);X=np.zeros((n,len(names)));rows=[None]*n;keep=np.zeros(n,bool)
    generated=np.zeros(n,bool);source=np.array(['']*n,dtype=object);target=np.array(['']*n,dtype=object);gain=np.zeros(n)
    for k,(tid,idx) in enumerate(bytarget.items()):
        t=d['targets'][tid];doc=byid[t['context_id']];tok=target_token(doc,t['target'],t['offset'])
        tg=resolve_target(tok)
        rel={a.lower():b for a,b in relation_map(tg[0],tg[1]).items()} if tg else {}
        cands=[];use=[]
        for i in idx:
            sub=d['substitutes'][records[i]['id']]['substitute'];key=sub.lower().replace(' ','_')
            lemma=(wn.morphy(key,tg[1]) if tg else None) or key
            hit=rel.get(key) or rel.get(lemma.lower())
            if split=='test' and not hit:continue
            cands.append({'word':sub,'lemma':lemma.replace('_',' '),'senses':list(hit[1]) if hit else [],'source':hit[0] if hit else None,
                          'multiword':' ' in sub,'pos_fallback':bool(tg and tg[3])})
            use.append(i)
        if not use:continue
        slot=E.Slot(doc,tok,target=tg)
        X[use]=stacker.features(slot,cands);vr=E.vector_rows(slot,cands)
        for j,i in enumerate(use):rows[i]=vr[j]
        keep[use]=True;generated[use]=[c['source'] is not None for c in cands];source[use]=[c['source'] or '' for c in cands]
        target[use]=tid;dd=difficulty.score([tok.text]+[c['word'] for c in cands]);gain[use]=dd[0]-dd[1:]
        if k%100==0:print('SWORDS',split,k,'of',len(bytarget),flush=True)
    out=dict(X=X,rows=rows,keep=keep,y=np.asarray(y,bool),groups=np.asarray(groups),generated=generated,source=source,target=target,gain=gain)
    pickle.dump(out,open(path,'wb'));return out

def simulate(p,y,target,gain,mask,cost,source=None):
    """Per target, apply the best candidate with positive utility, as the pipeline does.

    With source given (tiered), synonyms are preferred; other relations are used only when no synonym qualifies.
    """
    best={}
    for i in np.flatnonzero(mask&(gain>0)):
        u=utility(p[i],gain[i],cost)
        key=(-int(source[i]!='synonym') if source is not None else 0,u)
        if u>0 and (target[i] not in best or key>best[target[i]][0]):best[target[i]]=(key,i)
    correct=sum(bool(y[i]) for u,i in best.values())
    return {'error_cost':float(cost),'tiered':source is not None,'targets':int(len(set(target[mask]))),'edits':len(best),'correct':correct,'wrong':len(best)-correct}

dev=pairs('dev');assert dev['keep'].all()
X,y,g,gen=dev['X'],dev['y'],dev['groups'],dev['generated']
folds=list(GroupKFold(n_splits=5).split(X,y,g))
# Cross-fitting: each dev pair's checker score comes from a checker that never saw its context.
for a,b in folds:
    X[b,CHECK]=checker_model().fit([dev['rows'][i] for i in a],y[a]).predict_proba([dev['rows'][i] for i in b])[:,1]
checker=checker_model().fit(dev['rows'],y);pickle.dump(checker,open(RUN/'checker.pkl','wb'))
print('CHECKER cross-fitted AUC on generated pairs',round(roc_auc_score(y[gen],X[gen,CHECK]),4),flush=True)
search=[]
for C in [.03,.1,.3,1.,3.]:
    oof=np.zeros(len(y))
    for a,b in folds:oof[b]=E.StackedFit(members,C).fit(X[a],y[a]).proba(X[b])
    search.append({'C':C,'oof':oof,'auc_generated':float(roc_auc_score(y[gen],oof[gen]))})
chosen=max(search,key=lambda r:r['auc_generated']);oof=chosen['oof']
costs=[0.,.01,.02,.03,.05,.075,.1,.15,.2,.3]
curve=[simulate(oof,y,dev['target'],dev['gain'],gen,c,src) for src in [None,dev['source']] for c in costs]
# An edit should be right more often than wrong: maximize correct minus wrong, then prefer more edits.
operating=max(curve,key=lambda r:(r['correct']-r['wrong'],r['edits']))
stacker.C=chosen['C'];stacker.fit(X,y)
config.update(error_cost=operating['error_cost'],tiered=operating['tiered'],stacker_C=chosen['C'])
stacker.members=None;pickle.dump(stacker,open(RUN/'stacker.pkl','wb'));stacker.members=members
(RUN/'config.json').write_text(json.dumps(config,indent=2))
by_source={s:{'pairs':int((dev['source']==s).sum()),'positive_rate':float(y[dev['source']==s].mean())} for s in ['synonym','hypernym','similar']}
selection={'members':names,'weights':stacker.weights(),'C_search':[{k:v for k,v in r.items() if k!='oof'} for r in search],
           'operating_point':operating,'curve':curve,'dev_pairs':int(len(y)),'generated_pairs':int(gen.sum()),'contexts':int(len(set(g))),
           'by_source':by_source,'checker_crossfit_auc_generated':float(roc_auc_score(y[gen],X[gen,CHECK])),
           'criterion':'5-fold grouped CV over all SWORDS dev contexts with a cross-fitted checker; C by generated-pair AUC; error_cost by per-target correct minus wrong edits'}
(RUN/'fit_selection.json').write_text(json.dumps(selection,indent=2))
print('FROZEN',json.dumps({k:selection[k] for k in ['weights','operating_point','generated_pairs','by_source']}),flush=True)

# ---- Frozen. Reused test sets below are exploratory. ----
test=pairs('test');m=test['keep'].copy()
Xt=test['X'];Xt[m,CHECK]=checker.predict_proba([test['rows'][i] for i in np.flatnonzero(m)])[:,1]
p=np.zeros(len(m));p[m]=stacker.proba(Xt[m]);yt=test['y']
syn=m&(test['source']=='synonym')
swords={'generated':{'auc':float(roc_auc_score(yt[m],p[m])),**simulate(p,yt,test['target'],test['gain'],m,config['error_cost'],test['source'] if config['tiered'] else None)},
        'synonym_only_auc':float(roc_auc_score(yt[syn],p[syn])),
        'by_source':{s:{'pairs':int((m&(test['source']==s)).sum()),'positive_rate':float(yt[m&(test['source']==s)].mean())} for s in ['synonym','hypernym','similar']},
        'all_costs':[simulate(p,yt,test['target'],test['gain'],m,c,src) for src in [None,test['source']] for c in costs],
        'members_auc':{n:float(roc_auc_score(yt[m],Xt[m][:,j])) for j,n in enumerate(names) if np.std(Xt[m][:,j])>0}}
print('SWORDS TEST',json.dumps({k:v for k,v in swords.items() if k!='all_costs'}),flush=True)

pipeline=EnsembleClearText.load(RUN)
rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test')
outputs=[];lines=[]
for row,doc in zip(rows,docs):
    tok=target_token(doc,row['target']);rank=pipeline.rank_word(doc,tok)
    sel=rank['selected'];answer=sel['word'] if sel else row['target']
    outputs.append({'id':row['id'],'original':row['text'],'target':row['target'],'replacement':answer,'output':sel['output'] if sel else row['text'],
                    'changed':sel is not None,'gold_match':bool(sel and answer.lower() in row['gold']),'gold':row['gold'][:5],
                    'source':sel['source'] if sel else None,'senses':rank['senses'],
                    'candidates':[{k:c[k] for k in ['word','source','gain','fit','utility','rejections']} for c in rank['candidates']]})
    lines.append(row['text']+'\t'+row['target']+'\t'+answer.lower())
path=RUN/'tsar_ensemble_v2.tsv';path.write_text('\n'.join(lines)+'\n')
subprocess.run([sys.executable,str(RAW/'tsar_eval.py'),'--gold_file',str(RAW/'tsar_test.tsv'),'--predictions_file',str(path),'--output_file',str(RUN/'tsar_ensemble_v2_official.txt')],check=True,capture_output=True)
def summary(out):
    e=sum(r['changed'] for r in out);c=sum(r['gold_match'] for r in out)
    return {'n':len(out),'edits':e,'correct':c,'match_among_edits':c/e if e else 0.,'wrong_edits':e-c,
            'gold_generated':sum(bool(set(row['gold'])&{x['word'].lower() for x in r['candidates']}) for row,r in zip(rows,out)),
            'no_candidates':sum(not r['candidates'] for r in out)}
old=json.loads((ROOT/'runs/semcor-20260924/evaluation.json').read_text())['outputs']
v1out=json.loads((RUN_V1/'evaluation.json').read_text())['tsar_outputs']
tsar={'dictionary_basic':summary(old['basic']),'previous_sense_and_guards':summary(old['sense_and_guards']),
      'ensemble_v1':summary(v1out),'ensemble_v2':summary(outputs)}
tsar['v2_correct_by_source']={s:sum(r['gold_match'] and r['source']==s for r in outputs) for s in ['synonym','hypernym','similar']}
tsar['v2_edits_by_source']={s:sum(r['changed'] and r['source']==s for r in outputs) for s in ['synonym','hypernym','similar']}
sweep=[]
for cost in costs:
    e=c=0
    for row,r in zip(rows,outputs):
        ok=[x for x in r['candidates'] if x['gain']>0 and utility(x['fit'],x['gain'],cost)>0]
        if ok:
            best=max(ok,key=lambda x:(-int(config['tiered'] and x['source']!='synonym'),utility(x['fit'],x['gain'],cost)))
            e+=1;c+=best['word'].lower() in row['gold']
    sweep.append({'error_cost':cost,'edits':e,'correct':c})
tsar['diagnostic_sweep']=sweep  # Guardrails not rerun; not used for selection.
print('TSAR',json.dumps({k:v for k,v in tsar.items() if k!='diagnostic_sweep'}),flush=True)
print('SWEEP',json.dumps(sweep),flush=True)

examples=[r['original'] for r in json.loads((ROOT/'runs/initial-20260924/basic_examples_20.json').read_text())]
examples+=['The contract stipulates that payment is due monthly.','The gene encodes a protein that regulates growth.',
           'It was an auspicious start to the season.']
prev={r['original']:r['output'] for r in json.loads((RUN_V1/'evaluation.json').read_text())['samples']}
samples=[{'v1':prev.get(s),**pipeline.analyze(s,structure=False,phrases=False)} for s in examples]
for s in samples:print('SAMPLE',s['original'],'=>',s['output'],'| v1:',s['v1'],flush=True)
(RUN/'evaluation.json').write_text(json.dumps({'swords_test':swords,'tsar':tsar,'tsar_outputs':outputs,'samples':samples,
    'note':'SWORDS test and TSAR test were used in earlier experiments; exploratory only. No labels from them fit or selected any model.',
    'runtime_seconds':time.time()-started},indent=2,default=str))
print('COMPLETE',flush=True)
