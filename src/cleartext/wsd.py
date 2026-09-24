"""Supervised, non-transformer word sense disambiguation trained on SemCor."""
import math,re,pickle,json
from collections import Counter,defaultdict
from functools import lru_cache
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import SnowballStemmer
from .lexical import POS,synsets,stop
from .semantics import vectors
from .data import ROOT

stemmer=SnowballStemmer('english')
@lru_cache(100000)
def stem(word):return stemmer.stem(word.lower())
def context_features(words,index):
    # No target word or target label appears among context features.
    f=Counter()
    for i,w in enumerate(words):
        if i!=index and w.isalpha() and w.lower() not in stop():
            f['bag:'+stem(w)]+=1
            if abs(i-index)<=4:f['near:'+stem(w)]+=1
    for delta in [-2,-1,1,2]:
        j=index+delta
        if 0<=j<len(words):f[f'position{delta}:'+words[j].lower()]+=1
    return f
def context_embedding(words,index):
    vocab=vectors().vocab
    vs=[vocab[w.lower()].vector for i,w in enumerate(words)
        if i!=index and w.isalpha() and w.lower() not in stop() and vocab[w.lower()].has_vector]
    x=np.mean(vs,axis=0) if vs else np.zeros(300,dtype=np.float32)
    n=np.linalg.norm(x)
    return x/n if n else x
def word_pos(pos):return 'a' if pos=='s' else pos

class SenseModel:
    def __init__(self):
        self.counts={};self.words={};self.totals={};self.vocab={};self.centroids={}
        self.params={'nb':0.,'vector':0.}
    def fit(self,records):
        counts=defaultdict(Counter);words=defaultdict(lambda:defaultdict(Counter))
        sums={};voc=defaultdict(set)
        for r in records:
            key=(r['lemma'],r['pos']);sense=r['sense']
            f=context_features(r['words'],r['index'])
            counts[key][sense]+=1;words[key][sense].update(f);voc[key].update(f)
            x=context_embedding(r['words'],r['index'])
            k=(key,sense)
            if k not in sums:sums[k]=np.zeros(300,dtype=np.float64)
            sums[k]+=x
        self.counts=dict(counts)
        self.words={k:dict(v) for k,v in words.items()}
        self.totals={k:{s:sum(v.values()) for s,v in d.items()} for k,d in self.words.items()}
        self.vocab=dict(voc)
        self.centroids={k:(v/np.linalg.norm(v)).astype(np.float32) if np.linalg.norm(v) else v.astype(np.float32) for k,v in sums.items()}
        return self
    def components(self,words,index,lemma,pos):
        key=(lemma.lower(),word_pos(pos));senses=synsets(*key)
        if not senses:return []
        counts=self.counts.get(key,{});n=sum(counts.values())
        features=context_features(words,index)
        known={f:c for f,c in features.items() if f in self.vocab.get(key,set())}
        x=context_embedding(words,index);V=max(1,len(self.vocab.get(key,set())))
        out=[]
        for sense in senses:
            name=sense.name();prior=math.log((counts.get(name,0)+.5)/(n+.5*len(senses)))
            wc=self.words.get(key,{}).get(name,{})
            total=self.totals.get(key,{}).get(name,0)
            nb=sum(c*math.log((wc.get(f,0)+.5)/(total+.5*V)) for f,c in known.items())
            # Average evidence prevents longer sentences from overwhelming the prior.
            nb/=max(1,sum(known.values()))
            centroid=self.centroids.get((key,name))
            cos=float(x@centroid) if centroid is not None else 0.
            out.append({'sense':name,'prior':prior,'nb':nb,'vector':cos,
                        'training_examples':counts.get(name,0),'lemma_examples':n})
        return out
    def rank_components(self,rows,params=None):
        params=params or self.params
        scored=[{**r,'score':r['prior']+params['nb']*r['nb']+params['vector']*r['vector']} for r in rows]
        return sorted(scored,key=lambda r:-r['score'])
    def predict(self,words,index,lemma,pos):
        rows=self.rank_components(self.components(words,index,lemma,pos))
        if not rows:return {'sense':None,'margin':0,'ranked':[],'trained':False}
        return {'sense':rows[0]['sense'],'margin':rows[0]['score']-rows[1]['score'] if len(rows)>1 else 100.,
                'ranked':rows,'trained':rows[0]['lemma_examples']>0}
    def from_token(self,doc,token):
        # Tokenize consistently with SemCor words: punctuation is already tokenized by spaCy.
        return self.predict([t.text for t in doc],token.i,token.lemma_.lower(),POS.get(token.pos_,'n'))
    @classmethod
    def load(cls,run=None):
        run=run or ROOT/'runs/semcor-20260924'
        with open(run/'sense_model.pkl','rb') as f:return pickle.load(f)
