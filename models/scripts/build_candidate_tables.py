"""Cache every candidate's frozen-model scores for BenchLS dev/holdout and TSAR test.

Decision experiments (experiments/refine_round1_2.py) then run offline on these tables. Nothing here is fitted.
"""
import sys,pickle,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wordfreq import zipf_frequency
from cleartext.data import ROOT,benchls_data,tsar_data
from cleartext.features import parse,target_token,word_features
from cleartext.lexical import guardrails
from cleartext import ensemble as E
import os
from cleartext.ensemble_pipeline import EnsembleClearText,substitute,RUN
# REFINE_RUN / REFINE_VERSION select which fit components build the tables (default: v2 run, tables v2).
RUN=ROOT/os.environ.get('REFINE_RUN',str(RUN.relative_to(ROOT)));VERSION=os.environ.get('REFINE_VERSION','v2')
started=time.time()
pipe=EnsembleClearText.load(RUN)
names=[m.name for m in pipe.fit.members]
load=lambda p:pickle.load(open(p,'rb'))
word_model=load(ROOT/'runs/initial-20260924/word_model.pkl')
context_model=load(ROOT/'runs/initial-20260924/context_model.pkl')

def context_difficulty(doc,token,words,lemmas):
    """Context-aware CompLex model: the candidate's own word features in the target's sentence position."""
    rows=[]
    for w,l in zip(words,lemmas):
        f=word_features(w,doc,token,context=True);f['lemma_zipf']=zipf_frequency(l.lower(),'en');rows.append(f)
    return context_model.predict(rows)

def table(name,rows,docs,tokens):
    path=ROOT/f'data/cache/refine-{VERSION}-{name}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    out=[]
    for k,(row,doc,tok) in enumerate(zip(rows,docs,tokens)):
        case={'id':row['id'],'text':row['text'],'target':row['target'],'gold':row['gold'],'candidates':[]}
        if tok is None:out.append(case);continue
        target,cands=pipe.generator.generate(tok)
        case['target_info']=target
        if cands:
            slot=E.Slot(doc,tok,target=target)
            X=pipe.fit.features(slot,cands);fit=pipe.fit.proba(X)
            words=[tok.text]+[c['word'] for c in cands];lemmas=[target[0] if target else tok.lemma_]+[c['lemma'] for c in cands]
            dw=word_model.predict([word_features(w) for w in words]);dc=context_difficulty(doc,tok,words,lemmas)
            for j,c in enumerate(cands):
                output=substitute(doc,tok,c['word'])
                g=guardrails(doc.text,output,doc)
                case['candidates'].append({'word':c['word'],'lemma':c['lemma'],'source':c['source'],'multiword':c['multiword'],
                    'pos_fallback':c['pos_fallback'],'x':X[j],'fit':float(fit[j]),
                    'gain_word':float(dw[0]-dw[j+1]),'gain_context':float(dc[0]-dc[j+1]),
                    'target_difficulty':float(dw[0]),'guard':g['pass'],'guard_failed':g['failed'],
                    'gold':c['word'].lower() in row['gold']})
        out.append(case)
        if k%100==0:print('TABLE',name,k,'of',len(rows),flush=True)
    pickle.dump((names,out),open(path,'wb'));return names,out

b=benchls_data()
for split in ['dev','holdout']:
    rows=b[split];docs=parse([r['text'] for r in rows],f'benchls-{split}')
    tokens=[]
    for r,d in zip(rows,docs):
        offset=sum(len(w)+1 for w in r['text'].split(' ')[:r['index']])
        tokens.append(target_token(d,r['target'],offset))
    table(f'benchls-{split}',rows,docs,tokens)
rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test')
table('tsar-test',rows,docs,[target_token(d,r['target']) for r,d in zip(rows,docs)])
print('COMPLETE',round(time.time()-started),'s',flush=True)
