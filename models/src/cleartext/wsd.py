"""Supervised, non-transformer word sense disambiguation trained on SemCor."""
import math,pickle
from collections import Counter,defaultdict
from functools import lru_cache
import numpy as np
from nltk.stem import SnowballStemmer
from .lexical import POS,synsets,stop
from .semantics import vectors,gloss
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
        self.sums={}
        # alpha and shrink default to the original behavior so older pickles rank identically.
        self.params={'nb':0.,'vector':0.,'alpha':.5,'shrink':0.}
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
        self.sums={k:v.astype(np.float32) for k,v in sums.items()}
        return self
    def centroid(self,key,name,shrink):
        # Pseudo-examples pull sparse senses toward their WordNet gloss. Unseen senses use the gloss
        # alone instead of scoring zero, which previously favored any sense with a single example.
        if not shrink:return self.centroids.get((key,name))
        v=self.sums.get((key,name))
        v=(np.zeros(300,dtype=np.float32) if v is None else v)+shrink*gloss(name)
        n=np.linalg.norm(v)
        return v/n if n else None
    def components(self,words,index,lemma,pos,shrinks=None):
        key=(lemma.lower(),word_pos(pos));senses=synsets(*key)
        if not senses:return []
        counts=self.counts.get(key,{});n=sum(counts.values())
        features=context_features(words,index)
        known={f:c for f,c in features.items() if f in self.vocab.get(key,set())}
        x=context_embedding(words,index);V=max(1,len(self.vocab.get(key,set())))
        shrinks=[self.params.get('shrink',0.)] if shrinks is None else shrinks
        out=[]
        for sense in senses:
            name=sense.name();prior=math.log((counts.get(name,0)+.5)/(n+.5*len(senses)))
            wc=self.words.get(key,{}).get(name,{})
            total=self.totals.get(key,{}).get(name,0)
            nb=sum(c*math.log((wc.get(f,0)+.5)/(total+.5*V)) for f,c in known.items())
            # Average evidence prevents longer sentences from overwhelming the prior.
            nb/=max(1,sum(known.values()))
            cos={}
            for k in shrinks:
                centroid=self.centroid(key,name,k)
                cos[k]=float(x@centroid) if centroid is not None else 0.
            out.append({'sense':name,'prior':prior,'nb':nb,'vector':cos[shrinks[0]],'vectors':cos,
                        'training_examples':counts.get(name,0),'lemma_examples':n,'lemma_senses':len(senses)})
        return out
    def rank_components(self,rows,params=None):
        params=params or self.params
        alpha=params.get('alpha',.5);shrink=params.get('shrink',0.)
        def prior(r):
            if alpha==.5 or 'lemma_senses' not in r:return r['prior']
            return math.log((r['training_examples']+alpha)/(r['lemma_examples']+alpha*r['lemma_senses']))
        vector=lambda r:r.get('vectors',{}).get(shrink,r['vector'])
        scored=[{**r,'score':prior(r)+params['nb']*r['nb']+params['vector']*vector(r)} for r in rows]
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
