"""Candidate generators: a fourth interchangeable role alongside difficulty, sense and fit.

Every candidate records its WordNet source and the target senses it was reached from, so fit members can
score hypernyms and similar-to words differently from synonyms. Generators propose; they never decide.
"""
from abc import ABC,abstractmethod
from functools import lru_cache
from lemminflect import getInflection
from nltk.corpus import wordnet as wn
from .lexical import POS,synsets

TAGS={'v':['VB','VBZ','VBD','VBG','VBN','VBP'],'n':['NN','NNS'],'a':['JJ','JJR','JJS'],'r':['RB','RBR','RBS']}
SOURCES=['synonym','hypernym','similar']

def resolve_target(token):
    """(lemma, WordNet POS, Penn tag, fallback flag) for a token.

    The tagger's reading is used when WordNet knows it. Otherwise other parts of speech are tried with
    WordNet's own lemmatizer, and the tag is recovered by re-inflecting the lemma to the surface form.
    Fixes tagging errors such as "encodes" read as a noun.
    """
    if token is None:return None
    pos=POS.get(token.pos_)
    if pos and synsets(token.lemma_.lower(),pos):return token.lemma_.lower(),pos,token.tag_,False
    text=token.text.lower()
    for p in ['v','n','a','r']:
        lemma=wn.morphy(text,p)
        if not lemma or not synsets(lemma,p):continue
        tag=next((t for t in TAGS[p] if text in [f.lower() for f in getInflection(lemma,tag=t) or ()]),TAGS[p][0])
        return lemma,p,tag,True
    return None

def hyphen_fragment(token):
    """Part of a hyphenated word ("re" in "re-enacting"); simplifying the fragment breaks the word."""
    doc=token.doc
    after=token.i+1<len(doc) and not token.whitespace_ and doc[token.i+1].text=='-'
    before=token.i>0 and doc[token.i-1].text=='-' and not doc[token.i-1].whitespace_
    return after or before

@lru_cache(20000)
def relation_map(lemma,pos):
    """{candidate lemma as spelled in WordNet: (source, target senses)}. A lemma reachable several ways keeps the closest source."""
    found={}
    def add(name,source,sense):
        # Keyed by the WordNet spelling, as lexical.generate does, so "Latinised" keeps its capital.
        if name.lower()==lemma:return
        entry=found.setdefault(name,[source,[]])
        if SOURCES.index(source)<SOURCES.index(entry[0]):entry[0]=source
        if sense not in entry[1]:entry[1].append(sense)
    for s in synsets(lemma,pos):
        for l in s.lemma_names():add(l,'synonym',s.name())
        for h in s.hypernyms()+s.verb_groups():
            for l in h.lemma_names():add(l,'hypernym',s.name())
        for h in s.similar_tos()+s.also_sees():
            for l in h.lemma_names():add(l,'similar',s.name())
    return {k:(v[0],tuple(v[1])) for k,v in found.items()}

def inflect(lemma,tag,original):
    words=lemma.split('_')
    form=getInflection(words[0],tag=tag) if tag else None
    if tag and not form:return None
    words[0]=form[0] if form else words[0]
    word=' '.join(words)
    if original.istitle():word=word[:1].upper()+word[1:]
    elif original.isupper():word=word.upper()
    return word

class CandidateGenerator(ABC):
    name='generator'
    @abstractmethod
    def generate(self,token):
        """Return (target, candidates). target is (lemma, pos, tag, fallback) or None."""

class WordNetGenerator(CandidateGenerator):
    """Candidates from the chosen WordNet sources. sources=('synonym',) with multiword=False and
    fallback=False reproduces lexical.generate."""
    def __init__(self,sources=('synonym',),multiword=False,fallback=False,name=None):
        self.sources,self.multiword,self.fallback=tuple(sources),multiword,fallback
        self.name=name or '+'.join(self.sources)+('+multiword' if multiword else '')+('+fallback' if fallback else '')
    def generate(self,token):
        if token is not None and hyphen_fragment(token):return None,[]
        target=resolve_target(token)
        if target is None or (target[3] and not self.fallback):return target,[]
        lemma,pos,tag,fallback=target
        out={}
        for name,(source,senses) in relation_map(lemma,pos).items():
            if source not in self.sources or not name.replace('_','').isalpha():continue
            if '_' in name and not self.multiword:continue
            word=inflect(name,tag,token.text)
            if not word or word.lower()==token.text.lower():continue
            out.setdefault(word.lower(),{'word':word,'lemma':name.replace('_',' '),'senses':list(senses),
                                         'source':source,'multiword':'_' in name,'pos_fallback':fallback})
        return target,list(out.values())

class GeneratorUnion(CandidateGenerator):
    """Union of member generators; the first member to propose a word keeps it."""
    name='generator_union'
    def __init__(self,members):self.members=members
    def generate(self,token):
        target=None;out={}
        for m in self.members:
            t,cands=m.generate(token);target=target or t
            for c in cands:out.setdefault(c['word'].lower(),c)
        return target,list(out.values())
