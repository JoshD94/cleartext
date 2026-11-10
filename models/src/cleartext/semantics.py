"""Static-vector context features. No transformer or generative model."""
from functools import lru_cache
import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from .lexical import synsets, POS, context_vector

@lru_cache(1)
def vectors():
    return spacy.load('en_core_web_md',exclude=['tok2vec','tagger','parser','attribute_ruler','lemmatizer','ner'])

@lru_cache(40000)
def embed(text):
    doc=vectors().make_doc(text.lower())
    vs=[t.vector for t in doc if t.is_alpha and not t.is_stop and t.has_vector]
    v=np.mean(vs,axis=0) if vs else np.zeros(300)
    norm=np.linalg.norm(v)
    return v/norm if norm else v

@lru_cache(20000)
def gloss(name):
    s=wn.synset(name)
    return embed(' '.join([s.definition(),*s.examples(),*[h.definition() for h in s.hypernyms()]]))

def context_text(doc,token):
    # Original sentence is retained for context even when later stages split it.
    return ' '.join(t.lemma_ for t in doc if t.is_alpha and not t.is_stop and t.i!=token.i)

def sense_scores(doc,token):
    v=embed(context_text(doc,token))
    scores=[{'sense':s.name(),'definition':s.definition(),'similarity':float(v@gloss(s.name()))}
            for s in synsets(token.lemma_.lower(),POS.get(token.pos_,'n'))]
    return sorted(scores,key=lambda r:-r['similarity'])

def vector_features(doc,token,candidate,base=None,senses=None):
    f=dict(base) if base is not None else context_vector(doc,token,candidate)
    senses=sense_scores(doc,token) if senses is None else senses
    cv=embed(context_text(doc,token));source=embed(token.lemma_);cand=embed(candidate['word'])
    shared=set(candidate.get('senses',[]))
    best=senses[0]['similarity'] if senses else 0
    match=max((s['similarity'] for s in senses if s['sense'] in shared),default=-1)
    f.update({'vector_source_candidate':float(source@cand),'vector_candidate_context':float(cand@cv),
              'vector_source_context':float(source@cv),'vector_context_delta':float(cand@cv-source@cv),
              'gloss_best':best,'gloss_candidate':match,'gloss_gap':best-match,
              'gloss_top_member':int(bool(senses and senses[0]['sense'] in shared)),
              'gloss_margin':best-senses[1]['similarity'] if len(senses)>1 else 0,
              'vector_candidate_missing':int(not cand.any())})
    return f
