import hashlib, re
from functools import lru_cache
import numpy as np
import spacy
from spacy.tokens import DocBin
from nltk.corpus import cmudict
from wordfreq import zipf_frequency
from .data import ROOT

@lru_cache(1)
def nlp():
    return spacy.load('en_core_web_sm')

@lru_cache(1)
def pronunciations():
    return cmudict.dict()

@lru_cache(100000)
def base_features(word):
    lower=word.lower()
    freq=zipf_frequency(lower,'en')
    phones=pronunciations().get(lower)
    if phones:
        syllables=min(sum(p[-1:].isdigit() for p in pron) for pron in phones)
    else:
        syllables=max(1,len(re.findall(r'[aeiouy]+',lower))-int(lower.endswith('e') and not lower.endswith(('le','ye'))))
    return {'zipf':freq,'length':sum(c.isalpha() for c in word),'syllables':syllables,'frequency_missing':float(freq==0),'syllables_estimated':float(not phones)}

def parse(texts,cache_name):
    key=hashlib.sha256('\n'.join(texts).encode()).hexdigest()[:12]
    p=ROOT/'data/cache'/f'{cache_name}-{key}.spacy'
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        return list(DocBin().from_disk(p).get_docs(nlp().vocab))
    docs=list(nlp().pipe(texts,batch_size=128,n_process=1))
    DocBin(docs=docs,store_user_data=False).to_disk(p)
    return docs

def target_token(doc,target,offset=None):
    if offset is not None:
        for t in doc:
            if t.idx <= offset < t.idx+len(t):
                return t
    matches=[t for t in doc if t.text.lower()==target.lower()]
    if matches:
        return matches[0]
    index=doc.text.lower().find(target.lower())
    if index>=0:
        for t in doc:
            if t.idx<=index<t.idx+len(t):
                return t
    return None

def word_features(word,doc=None,token=None,context=False):
    f=dict(base_features(word))
    if context:
        if token is not None:
            neighbors=[t for t in doc[max(0,token.i-3):token.i+4] if t.is_alpha and t.i!=token.i]
            f.update({'pos':token.pos_,'dependency':token.dep_,'lemma_zipf':zipf_frequency(token.lemma_.lower(),'en'),
                      'local_zipf':float(np.mean([base_features(t.text)['zipf'] for t in neighbors])) if neighbors else 0,
                      'sentence_length':sum(t.is_alpha for t in doc),'is_entity':float(bool(token.ent_type_))})
        else:
            f.update({'pos':'UNKNOWN','dependency':'UNKNOWN','lemma_zipf':f['zipf'],'local_zipf':0,'sentence_length':len(doc) if doc else 0,'is_entity':0})
    return f

def sentence_features(docs,word_model):
    lengths=[]; flat=[]
    for doc in docs:
        tokens=[t for t in doc if t.is_alpha]
        lengths.append(len(tokens))
        flat.extend(word_features(t.text) for t in tokens)
    predictions=[]
    for i in range(0,len(flat),20000):
        predictions.extend(np.clip(word_model.predict(flat[i:i+20000]),0,1))
    out=[]; start=0
    for doc,n in zip(docs,lengths):
        tokens=[t for t in doc if t.is_alpha]
        vals=np.asarray(predictions[start:start+n]); start+=n
        bf=[base_features(t.text) for t in tokens]
        depths=[]
        for t in tokens:
            cur=t; depth=0; seen=set()
            while cur.head.i!=cur.i and cur.i not in seen:
                seen.add(cur.i); depth+=1; cur=cur.head
            depths.append(depth)
        freq=[f['zipf'] for f in bf] or [0]
        f={'length':n,'word_mean':float(vals.mean()) if n else 0,'word_max':float(vals.max()) if n else 0,
           'word_p90':float(np.quantile(vals,.9)) if n else 0,'mean_zipf':float(np.mean(freq)),
           'rare_fraction':sum(v<3 for v in freq)/max(n,1),'mean_syllables':float(np.mean([x['syllables'] for x in bf])) if n else 0,
           'mean_word_length':float(np.mean([len(t) for t in tokens])) if n else 0,
           'max_depth':max(depths,default=0),'mean_dependency_distance':float(np.mean([abs(t.i-t.head.i) for t in tokens])) if n else 0,
           'subclauses':sum(t.dep_ in {'advcl','ccomp','xcomp','relcl','acl'} for t in doc),
           'coordination':sum(t.dep_=='conj' for t in doc),'passive':sum(t.dep_ in {'nsubjpass','auxpass'} for t in doc),
           'punctuation':sum(t.is_punct for t in doc),'lexical_density':sum(t.pos_ in {'NOUN','VERB','ADJ','ADV'} for t in tokens)/max(n,1)}
        out.append(f)
    return out
