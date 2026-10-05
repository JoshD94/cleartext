"""Hard-word detector (CWI 2018): feature rows, the cached CWI loader, and scoring.

The detector sees a word's context features, both CompLex difficulty scores, and a few token flags. The same
row builder serves training (scripts/train_detector.py), the pipeline, and the experiments.
"""
import csv,pickle
import numpy as np
from wordfreq import zipf_frequency
from .data import ROOT,RAW,cwi_download
from .features import parse,target_token,word_features

CONTENT={'NOUN','VERB','ADJ','ADV'}

def row(context_features,word_score,context_score,stop,entity,word):
    return {**context_features,'word_score':float(word_score),'context_score':float(context_score),
            'stop':float(stop),'entity':float(entity),'lower':float(word.islower())}

def rows_for(doc,tokens,word_scores,context_model,words=None,lemmas=None):
    """Detector rows for tokens in doc. With words/lemmas, each word is scored in the matching token's position
    (a candidate replacement read in the target's place); lemma frequency then comes from the given lemma."""
    words=words or [t.text for t in tokens]
    cf=[]
    for i,(t,w) in enumerate(zip(tokens,words)):
        f=word_features(w,doc,t,context=True)
        if lemmas is not None:f['lemma_zipf']=zipf_frequency(lemmas[i].lower(),'en')
        cf.append(f)
    cs=context_model.predict(cf) if cf else []
    return [row(f,a,b,t.is_stop,bool(t.ent_type_),w) for f,a,b,t,w in zip(cf,word_scores,cs,tokens,words)]

def probabilities(detector,rows):
    return detector['model'].predict_proba(rows)[:,1] if rows else np.array([])

def cwi_rows(split,word_model,context_model):
    """Single-word CWI 2018 English targets with label, annotator share, and precomputed features (cached)."""
    path=ROOT/f'data/cache/cwi-rows-v1-{split}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    cwi_download()
    raw=[r for r in csv.reader(open(RAW/f'cwi_english_{split}.tsv'),delimiter='\t',quoting=csv.QUOTE_NONE) if ' ' not in r[4].strip()]
    sents=sorted({r[1] for r in raw});docs=dict(zip(sents,parse(sents,f'cwi-{split}')))
    out=[]
    for r in raw:
        doc=docs[r[1]];tok=target_token(doc,r[4],int(r[2]))
        if tok is None:continue
        out.append({'label':r[9]=='1','prob':float(r[10]),'word':tok.text,'pos':tok.pos_,'stop':tok.is_stop,'entity':bool(tok.ent_type_),
                    'content':tok.pos_ in CONTENT and not tok.is_stop and not tok.ent_type_,
                    'wf':word_features(tok.text),'cf':word_features(tok.text,doc,tok,context=True)})
    ws=word_model.predict([r['wf'] for r in out]);cs=context_model.predict([r['cf'] for r in out])
    for r,a,b in zip(out,ws,cs):r['word_score'],r['context_score']=float(a),float(b)
    pickle.dump(out,open(path,'wb'));return out

def cwi_features(rows):
    return [row(r['cf'],r['word_score'],r['context_score'],r['stop'],r['entity'],r['word']) for r in rows]
