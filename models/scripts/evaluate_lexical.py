import sys,json,gzip,pickle,time,subprocess
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
from cleartext.data import ROOT,RAW,tsar_data
from cleartext.features import parse,target_token
from cleartext.lexical import generate,context_vector,LexicalPipeline,guardrails,synsets,POS,language_model

run=ROOT/'runs/initial-20260924';started=time.time()
def swords_features(split):
    cache=ROOT/f'data/cache/swords-features-{split}.pkl'
    if cache.exists():
        with open(cache,'rb') as f:return pickle.load(f)
    d=json.load(gzip.open(RAW/f'swords_{split}.json.gz','rt'))
    ids=list(d['contexts']);docs=parse([d['contexts'][i]['context'] for i in ids],f'swords-{split}');byid=dict(zip(ids,docs))
    feats=[];labels=[];groups=[];records=[];skip=0
    for sid,c in d['substitutes'].items():
        t=d['targets'][c['target_id']];doc=byid[t['context_id']];tok=target_token(doc,t['target'],t['offset'])
        votes=d['substitute_labels'][sid]
        if tok is None or not votes:skip+=1;continue
        # Same label convention as benchmark examples: TRUE votes divided by all judgments.
        proportion=votes.count('TRUE')/len(votes)
        candidate={'word':c['substitute'],'senses':[s.name() for s in synsets(tok.lemma_.lower(),POS.get(tok.pos_,'n')) if c['substitute'].lower().replace(' ','_') in {x.lower() for x in s.lemma_names()}]}
        feats.append(context_vector(doc,tok,candidate));labels.append(int(proportion>=.5));groups.append(t['context_id'])
        records.append({'id':sid,'target':t['target'],'candidate':c['substitute'],'human_fraction':proportion})
    result=(feats,np.asarray(labels),np.asarray(groups),records,skip)
    with open(cache,'wb') as f:pickle.dump(result,f)
    print('SWORDS features',split,len(labels),'positive',sum(labels),'skipped',skip,flush=True)
    return result

def clf_metrics(y,p,threshold):
    pred=p>=threshold
    return {'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),'auc':float(roc_auc_score(y,p)),'acceptance':float(pred.mean()),'n':len(y)}

language_model()
X,y,groups,records,skip=swords_features('dev')
train_idx,valid_idx=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=4701).split(X,y,groups))
results=[];fitted={};thresholds={}
for name,est in [('Logistic regression',LogisticRegression(C=1,max_iter=1000)),('Gradient boosting',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=10,random_state=4701))]:
    pipe=make_pipeline(DictVectorizer(sparse=False),StandardScaler(),est)
    pipe.fit([X[i] for i in train_idx],y[train_idx]);p=pipe.predict_proba([X[i] for i in valid_idx])[:,1]
    thresholds[name]=.5;fitted[name]=pipe
    score=clf_metrics(y[valid_idx],p,.5)
    results.append({'name':name,'dev':score});print('Context model',name,score,flush=True)
selected=max(results,key=lambda r:r['dev']['f1'])['name'];checker=fitted[selected]
pv=checker.predict_proba([X[i] for i in valid_idx])[:,1]
curve=[{'threshold':float(t),**clf_metrics(y[valid_idx],pv,t)} for t in np.arange(.1,.91,.05)]
eligible=[r for r in curve if r['precision']>=.75 and r['acceptance']>=.01]
operating=max(eligible,key=lambda r:r['recall']) if eligible else max(curve,key=lambda r:r['f1'])
threshold=operating['threshold']
(run/'fit_selection.json').write_text(json.dumps({'model':selected,'threshold':threshold,'criterion':'Dev F1 for model; maximum recall at >=75% precision for threshold, else max F1','dev_curve':curve},indent=2))
with open(run/'fit_model.pkl','wb') as f:pickle.dump(checker,f)
Xt,yt,gt,rt,skiptest=swords_features('test')
for r in results:r['test']=clf_metrics(yt,fitted[r['name']].predict_proba(Xt)[:,1],.5)
ptest=checker.predict_proba(Xt)[:,1]
context_result={'selected':selected,'threshold':threshold,'results':results,'operating_dev':operating,'operating_test':clf_metrics(yt,ptest,threshold),
 'curve':curve,'counts':{'train_pairs':len(train_idx),'validation_pairs':len(valid_idx),'test_pairs':len(yt),'train_contexts':len(set(groups[train_idx])),'validation_contexts':len(set(groups[valid_idx])),'test_contexts':len(set(gt))},'skipped':{'dev':skip,'test':skiptest},
 'label':'At least half of all judgments are TRUE. Custom binary diagnostic, not official SWORDS ranking scores.'}
with open(run/'word_model.pkl','rb') as f:word_model=pickle.load(f)
pipe=LexicalPipeline(word_model,{'min_gain':.02,'min_fit':threshold},checker)
rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test')
prepared=[(r,d,target_token(d,r['target'])) for r,d in zip(rows,docs)]
generated=[generate(d,t) for r,d,t in prepared]
pipe.prime([r['target'] for r in rows]+[c['word'] for cs in generated for c in cs])
modes=['unchanged','dictionary','learned','context','guarded'];outputs={m:[] for m in modes};mode_results=[]
coverage=sum(bool({c['word'].lower() for c in cs}&set(r['gold'])) for r,cs in zip(rows,generated))/len(rows)
for mode in modes:
    correct=changed=possible5=0;start=time.time();lines=[]
    for (r,d,t),cs in zip(prepared,generated):
        if mode=='unchanged':out={'original':r['text'],'target':r['target'],'replacement':r['target'],'output':r['text'],'changed':False,'candidates':[],'reason':'Unchanged baseline'}
        else:out=pipe.rank(d,t,cs,mode)
        gold=set(r['gold']);accepted=[c['word'].lower() for c in out['candidates'] if c['eligible']]
        if mode=='guarded':accepted=[out['replacement'].lower()] if out['changed'] else []
        answer=out['replacement'].lower() if out.get('replacement') else r['target'].lower()
        hit=bool(out['changed'] and answer in gold)
        correct+=int(hit);changed+=int(out['changed']);possible5+=int(bool(set(accepted[:5])&gold))
        out['id']=r['id'];out['gold']=list(dict.fromkeys(r['gold']));out['gold_match']=hit
        outputs[mode].append(out)
        predictions=([answer]+[c for c in accepted if c!=answer])[:10]
        lines.append(r['text']+'\t'+r['target']+'\t'+'\t'.join(predictions))
    path=run/f'tsar_{mode}_predictions.tsv';path.write_text('\n'.join(lines)+'\n')
    evalpath=run/f'tsar_{mode}_official.txt'
    result=subprocess.run([sys.executable,str(RAW/'tsar_eval.py'),'--gold_file',str(RAW/'tsar_test.tsv'),'--predictions_file',str(path),'--output_file',str(evalpath)],capture_output=True,text=True)
    official=evalpath.read_text() if evalpath.exists() else result.stdout+result.stderr
    if result.returncode:print('Official evaluator error',mode,result.stderr,flush=True)
    edits=[o for o in outputs[mode] if o['changed']]
    mode_results.append({'name':mode,'n':len(rows),'top1_gold_match':correct/len(rows),'edit_rate':changed/len(rows),
      'edited_gold_match':sum(o['gold_match'] for o in edits)/len(edits) if edits else None,'potential5':possible5/len(rows),
      'runtime_seconds':time.time()-start,'official_evaluator_exit':result.returncode,'official_output':official})
    print('TSAR',mode,mode_results[-1],flush=True)

# Authored diagnostic examples are never used to tune models or thresholds.
examples=[
 ('CS','The algorithm utilizes a neural network to classify images.','utilizes'),
 ('CS','The procedure commences after the server receives the request.','commences'),
 ('CS','The compiler eliminates redundant instructions.','redundant'),
 ('CS','The system may terminate the connection after 30 seconds.','terminate'),
 ('CS','The model did not improve the accuracy.','improve'),
 ('Corporate','The company purchased additional equipment.','purchased'),
 ('Corporate','The bank approved the loan yesterday.','bank'),
 ('Corporate','The firm anticipates a reduction in expenditure.','expenditure'),
 ('Corporate','The annual report disclosed a substantial loss.','substantial'),
 ('Corporate','The interest rate may increase by 2 percent.','increase'),
 ('Politics','The committee rejected the proposed legislation.','legislation'),
 ('Politics','The minister subsequently addressed the assembly.','subsequently'),
 ('Politics','The policy could exacerbate existing inequality.','exacerbate'),
 ('Politics','The proposal requires parliamentary approval.','requires'),
 ('Politics','The witness did not corroborate the allegation.','corroborate'),
 ('Ambiguity','They walked along the bank of the river.','bank'),
 ('Ambiguity','The committee will table the proposal tomorrow.','table'),
 ('Technical term','The neural network learns from examples.','neural'),
 ('Grammar','The researchers investigated the unusual result.','investigated'),
 ('Grammar','The results demonstrate a consistent pattern.','demonstrate')]
example_docs=parse([t for d,t,w in examples],'diagnostic')
review=[]
for (domain,text,target),doc in zip(examples,example_docs):
    token=target_token(doc,target);cs=generate(doc,token)
    versions={mode:pipe.rank(doc,token,cs,mode) for mode in ['dictionary','learned','context','guarded']}
    review.append({'id':f'authored-{len(review)}','domain':domain,**versions['guarded'],'versions':versions})
probes=[
 ('Number','The contract costs 50 dollars.','The contract costs 500 dollars.',False),
 ('Negation','The result is not significant.','The result is significant.',False),
 ('Modality','The firm may expand.','The firm will expand.',False),
 ('Entity roles','Alice defeated Bob.','Bob defeated Alice.',False),
 ('Technical phrase','The neural network learns.','The nerve network learns.',False),
 ('Unit','The package weighs 5 kilograms.','The package weighs 5 grams.',False),
 ('Antonym','The firm approved the proposal.','The firm rejected the proposal.',False),
 ('Specificity','The dog barked.','The animal barked.',False),
 ('Identity','The bank approved a loan.','The bank approved a loan.',True),
 ('Safe synonym','They purchased equipment.','They bought equipment.',True)]
probe_results=[{'category':c,'original':a,'candidate':b,'expected_pass':e,**guardrails(a,b)} for c,a,b,e in probes]
result={'context_fit':context_result,'tsar':mode_results,'candidate_coverage':coverage,'outputs':outputs,'review':review,
        'guardrail_probes':probe_results,'duration_seconds':time.time()-started,
        'settings':pipe.settings,'protected_terms_source':'Small hand-authored technical phrase list; not a complete jargon detector',
        'scope':'Single target, single-word edits. No sentence restructuring. TSAR supplies the difficult target; full detection is evaluated separately.'}
(run/'lexical.json').write_text(json.dumps(result,indent=2))
print('LEXICAL_COMPLETE',round(time.time()-started,1),flush=True)
