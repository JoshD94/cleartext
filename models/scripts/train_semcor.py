import sys,json,pickle,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import nltk
from nltk.corpus import semcor
from nltk.tree import Tree
from cleartext.data import ROOT
from cleartext.wsd import SenseModel,word_pos,synsets
run=ROOT/'runs/semcor-20260924';run.mkdir(exist_ok=True)
started=time.time()
cache=ROOT/'data/cache/semcor-records-v1.pkl'
if cache.exists():
    data,manifest=pickle.load(open(cache,'rb'))
else:
    files=sorted(semcor.fileids());random.Random(4701).shuffle(files)
    n=len(files);assigned={f:('train' if i<int(.7*n) else 'dev' if i<int(.85*n) else 'test') for i,f in enumerate(files)}
    data={s:[] for s in ['train','dev','test']};manifest={'documents':{s:[] for s in data},'skipped':{},'source':'NLTK SemCor, WordNet 3.0 mappings'}
    skipped={}
    for fi,f in enumerate(files):
        split=assigned[f];manifest['documents'][split].append(f)
        for si,sent in enumerate(semcor.tagged_sents(f,tag='sem')):
            words=[];annotations=[]
            for chunk in sent:
                leaves=chunk.leaves() if isinstance(chunk,Tree) else list(chunk)
                start=len(words);words+=leaves
                if not isinstance(chunk,Tree):continue
                label=chunk.label()
                if not hasattr(label,'synset'):continue
                if len(leaves)!=1 or not leaves[0].isalpha():
                    skipped['multiword_or_nonalpha']=skipped.get('multiword_or_nonalpha',0)+1;continue
                if any(isinstance(x,Tree) and x.label()=='NE' for x in chunk):
                    skipped['named_entity']=skipped.get('named_entity',0)+1;continue
                sense=label.synset();lemma=label.name().lower();pos=word_pos(sense.pos())
                if '_' in lemma or sense.name() not in [s.name() for s in synsets(lemma,pos)]:
                    skipped['unmatched_lemma']=skipped.get('unmatched_lemma',0)+1;continue
                annotations.append((start,lemma,pos,sense.name()))
            textkey=hashlib.sha256(' '.join(words).lower().encode()).hexdigest()
            for index,lemma,pos,sense in annotations:
                data[split].append({'document':f,'sentence':si,'sentence_hash':textkey,'words':words,'index':index,'lemma':lemma,'pos':pos,'sense':sense})
        if fi%50==0:print('READ',fi+1,'of',n,flush=True)
    # Keep evaluation sentences out of earlier splits, including exact repeats across documents.
    testhash={r['sentence_hash'] for r in data['test']};devhash={r['sentence_hash'] for r in data['dev']}
    removed={}
    for split,excluded in [('train',testhash|devhash),('dev',testhash)]:
        before=len(data[split]);data[split]=[r for r in data[split] if r['sentence_hash'] not in excluded];removed[split]=before-len(data[split])
    manifest.update(skipped=skipped,duplicate_annotations_removed=removed)
    zip_path=Path(str(nltk.data.find('corpora/semcor.zip')))
    manifest['sha256']=hashlib.sha256(zip_path.read_bytes()).hexdigest()
    pickle.dump((data,manifest),open(cache,'wb'))
manifest['counts']={s:len(v) for s,v in data.items()}
(run/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COUNTS',manifest['counts'],flush=True)
model=SenseModel().fit(data['train'])
print('FIT',len(model.counts),'lemma/POS groups',flush=True)
def components(split):
    path=ROOT/f'data/cache/semcor-components-v1-{split}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    out=[model.components(r['words'],r['index'],r['lemma'],r['pos']) for r in data[split]]
    pickle.dump(out,open(path,'wb'));return out
def measure(split,comp,params):
    predictions=[model.rank_components(c,params)[0]['sense'] for c in comp]
    hits=np.array([a==r['sense'] for a,r in zip(predictions,data[split])])
    ambiguous=np.array([len(c)>1 for c in comp])
    seen=np.array([c[0]['lemma_examples']>0 for c in comp])
    return {'accuracy':float(hits.mean()),'ambiguous_accuracy':float(hits[ambiguous].mean()),
            'ambiguous_n':int(ambiguous.sum()),'seen_lemma_accuracy':float(hits[seen].mean()),
            'seen_lemma_fraction':float(seen.mean()),'n':len(hits)}
dc=components('dev')
results=[]
for nb in [0.,1.,3.,6.]:
    for vector in [0.,2.,5.]:
        params={'nb':nb,'vector':vector}
        metrics=measure('dev',dc,params)
        results.append({'params':params,**metrics})
best=max(results,key=lambda r:r['accuracy'])
model.params=best['params']
selection={'params':model.params,'development':results,'selected_dev':best,'description':'Maximum document-held-out development WSD accuracy; no SWORDS/TSAR scores used to fit or select this model.'}
(run/'selection.json').write_text(json.dumps(selection,indent=2))
pickle.dump(model,open(run/'sense_model.pkl','wb'))
tc=components('test')
report={'selection':selection,'test':measure('test',tc,model.params),
        'baseline_test':measure('test',tc,{'nb':0.,'vector':0.}),
        'counts':manifest['counts'],'runtime_seconds':time.time()-started,
        'scope':'Custom document-held-out SemCor split. Gold target lemma/POS supplied. Not an official WSD benchmark; parsed deployment inputs introduce extra errors.'}
(run/'metrics.json').write_text(json.dumps(report,indent=2))
print('RESULT',json.dumps({k:v for k,v in report.items() if k!='selection'}),flush=True)
print('SELECTED',model.params,flush=True)
