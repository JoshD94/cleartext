"""Stage 1: refit the SemCor sense model with gloss backoff, then tune the sense ensemble on SemCor dev only."""
import sys,json,pickle,time,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from cleartext.data import ROOT
from cleartext.wsd import SenseModel,context_embedding
from cleartext.semantics import gloss
run=ROOT/'runs/ensemble-20260927';run.mkdir(exist_ok=True)
started=time.time()
data,manifest=pickle.load(open(ROOT/'data/cache/semcor-records-v1.pkl','rb'))
model=SenseModel().fit(data['train'])
print('FIT',len(model.counts),'lemma/POS groups',flush=True)
SHRINKS=[0.,1.,3.,10.]

def table(split):
    """Flat arrays, one row per (annotation, candidate sense). Gold senses outside WordNet rows are kept as misses."""
    path=ROOT/f'data/cache/ensemble-senses-v1-{split}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    rows=[];group=[];gold=[]
    for i,r in enumerate(data[split]):
        comp=model.components(r['words'],r['index'],r['lemma'],r['pos'],shrinks=SHRINKS)
        x=context_embedding(r['words'],r['index'])
        for c in comp:
            rows.append([c['training_examples'],c['lemma_examples'],c['lemma_senses'],c['nb'],float(x@gloss(c['sense'])),*[c['vectors'][k] for k in SHRINKS]])
            group.append(i);gold.append(c['sense']==r['sense'])
        if i%5000==0:print('COMPONENTS',split,i,flush=True)
    out=(np.array(rows),np.array(group),np.array(gold),len(data[split]))
    pickle.dump(out,open(path,'wb'));return out

def group_softmax(score,group,temperature=1.):
    s=score/temperature
    m=np.full(group.max()+1,-np.inf);np.maximum.at(m,group,s)
    e=np.exp(s-m[group]);z=np.zeros(group.max()+1);np.add.at(z,group,e)
    return e/z[group]

def evaluate(prob,group,gold,n):
    """Accuracy over all annotations and mean gold log-likelihood. Ties go to the earlier WordNet sense, as in rank_components."""
    order=np.lexsort((np.arange(len(prob)),-prob,group))
    first=order[np.r_[True,group[order][1:]!=group[order][:-1]]]
    hit=np.zeros(n,bool);hit[group[first]]=gold[first]
    ll=np.full(n,np.log(1e-6));ll[group[gold]]=np.log(np.maximum(prob[gold],1e-6))
    return {'accuracy':float(hit.mean()),'log_likelihood':float(ll.mean()),'n':n}

def semcor_score(X,params):
    count,n,k,nb=X[:,0],X[:,1],X[:,2],X[:,3]
    prior=np.log((count+params['alpha'])/(n+params['alpha']*k))
    return prior+params['nb']*nb+params['vector']*X[:,5+SHRINKS.index(params['shrink'])]

Xd,gd,yd,nd=table('dev')
grid=[]
for alpha,shrink,nb,vector in itertools.product([.5,1.,2.],SHRINKS,[0.,1.,3.,6.],[0.,2.,5.,8.]):
    params={'alpha':alpha,'shrink':shrink,'nb':nb,'vector':vector}
    grid.append({'params':params,**evaluate(group_softmax(semcor_score(Xd,params),gd),gd,yd,nd)})
best=max(grid,key=lambda r:r['accuracy'])['params']
print('SEMCOR PARAMS',best,flush=True)
temps=[.25,.5,.75,1.,1.5,2.,3.]
semcor_t=max(temps,key=lambda t:evaluate(group_softmax(semcor_score(Xd,best),gd,t),gd,yd,nd)['log_likelihood'])
gloss_t=max([.02,.03,.05,.08,.12,.2],key=lambda t:evaluate(group_softmax(Xd[:,4],gd,t),gd,yd,nd)['log_likelihood'])
ps=group_softmax(semcor_score(Xd,best),gd,semcor_t);pg=group_softmax(Xd[:,4],gd,gloss_t)
def mixture(w,evidence,X,ps,pg):
    n=X[:,1];keep=w*n/(n+evidence) if evidence else np.full(len(n),w)
    return keep*ps+(1-keep)*pg
mix=[{'semcor_weight':float(w),'evidence':ev,**evaluate(mixture(w,ev,Xd,ps,pg),gd,yd,nd)}
     for w in np.round(np.arange(0,1.01,.1),2) for ev in [0.,1.,3.,10.,30.]]
# Likelihood picks the weights because the fit scorers consume probabilities, not only the top sense.
chosen=max(mix,key=lambda r:r['log_likelihood']);w,ev=chosen['semcor_weight'],chosen['evidence']
model.params=best
config={'semcor_params':best,'semcor_temperature':semcor_t,'gloss_temperature':gloss_t,'sense_weights':[float(w),float(1-w)],'sense_evidence':ev}
pickle.dump(model,open(run/'sense_model.pkl','wb'))
(run/'sense_selection.json').write_text(json.dumps({**config,'grid':grid,'mixture':mix,
    'criterion':'SemCor document-held-out dev: accuracy for SemCor parameters, log-likelihood for temperatures and mixture weight'},indent=2))
# Selection frozen. Held-out SemCor test below.
Xt,gt,yt,nt=table('test')
old={'alpha':.5,'shrink':0.,'nb':3.,'vector':5.}
pt_s=group_softmax(semcor_score(Xt,best),gt,semcor_t);pt_g=group_softmax(Xt[:,4],gt,gloss_t)
test={'frequency_baseline':evaluate(group_softmax(semcor_score(Xt,{**old,'nb':0.,'vector':0.}),gt),gt,yt,nt),
      'previous_semcor_model':evaluate(group_softmax(semcor_score(Xt,old),gt),gt,yt,nt),
      'semcor_with_backoff':evaluate(pt_s,gt,yt,nt),'gloss_only':evaluate(pt_g,gt,yt,nt),
      'sense_ensemble':evaluate(mixture(w,ev,Xt,pt_s,pt_g),gt,yt,nt)}
unseen=np.zeros(nt,bool);unseen[gt[Xt[:,1]==0]]=True
test['sense_ensemble_unseen_lemmas']=evaluate(mixture(w,ev,Xt,pt_s,pt_g)[unseen[gt]],np.unique(gt[unseen[gt]],return_inverse=True)[1],yt[unseen[gt]],int(unseen.sum()))
test['previous_unseen_lemmas']=evaluate(group_softmax(semcor_score(Xt,old),gt)[unseen[gt]],np.unique(gt[unseen[gt]],return_inverse=True)[1],yt[unseen[gt]],int(unseen.sum()))
(run/'sense_metrics.json').write_text(json.dumps({**config,'test':test,'runtime_seconds':time.time()-started,
    'scope':'Custom document-held-out SemCor split, gold lemma/POS supplied. The previous model is reproduced from the same refit counts.'},indent=2))
(run/'config.json').write_text(json.dumps(config,indent=2))
print('TEST',json.dumps(test),flush=True)
print('COMPLETE',flush=True)
