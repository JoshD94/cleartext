"""Interchangeable model roles and ensembles.

Every role has a small abstract interface. Each ensemble implements the same interface as its members,
so one model or a combination can fill a role without changing the pipeline. No model here rewrites text;
the pipeline in ensemble_pipeline.py decides edits from their scores.
"""
from abc import ABC,abstractmethod
from dataclasses import dataclass,field
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .features import word_features,base_features
from .lexical import POS,synsets,local_lm
from .semantics import embed,gloss,vector_features,sense_scores
from .wsd import word_pos,context_embedding

def softmax(scores,temperature=1.):
    x=np.asarray(scores,dtype=float)/max(temperature,1e-6)
    x=np.exp(x-x.max())
    return x/x.sum()

@dataclass
class Slot:
    """One target word in context. The context doc is the original input, before earlier rewrites."""
    doc:object
    token:object
    context_doc:object=None
    context_token:object=None
    cache:dict=field(default_factory=dict)
    target:tuple=None  # (lemma, WordNet POS, tag, fallback) from generation.resolve_target; overrides the tagger
    def __post_init__(self):
        if self.context_doc is None:self.context_doc,self.context_token=self.doc,self.token
    @property
    def words(self):return [t.text for t in self.context_doc]
    @property
    def index(self):return self.context_token.i
    @property
    def lemma(self):return self.target[0] if self.target else self.context_token.lemma_.lower()
    @property
    def pos(self):return self.target[1] if self.target else POS.get(self.context_token.pos_,'n')
    def substituted(self,candidate):
        """Context words with the candidate in place, for round-trip sense checks."""
        w=self.words;w[self.index]=candidate['word'];return w

# Difficulty: word -> [0, 1] ---------------------------------------------------------------

class DifficultyModel(ABC):
    name='difficulty'
    @abstractmethod
    def score(self,words):...

class RegressorDifficulty(DifficultyModel):
    """Wraps a frozen CompLex regressor (models.Scorer) with a per-word cache."""
    def __init__(self,scorer,name='complex_word'):
        self.scorer,self.name,self.cache=scorer,name,{}
    def score(self,words):
        missing=sorted({w.lower() for w in words}-self.cache.keys())
        if missing:
            for w,p in zip(missing,self.scorer.predict([word_features(w) for w in missing])):self.cache[w]=float(p)
        return np.array([self.cache[w.lower()] for w in words])

class DifficultyEnsemble(DifficultyModel):
    name='difficulty_ensemble'
    def __init__(self,members,weights=None):
        self.members=members;self.weights=np.ones(len(members))/len(members) if weights is None else np.asarray(weights)
    def score(self,words):
        return sum(w*m.score(words) for m,w in zip(self.members,self.weights))

# Sense: context -> probability over the target's WordNet senses ----------------------------

class SenseDistribution(ABC):
    name='sense'
    @abstractmethod
    def distribution(self,words,index,lemma,pos):
        """Return {synset name: probability}; empty when the lemma has no WordNet senses."""

class SemCorSense(SenseDistribution):
    name='semcor'
    def __init__(self,model,temperature=1.):self.model,self.temperature=model,temperature
    def examples(self,lemma,pos):return sum(self.model.counts.get((lemma.lower(),word_pos(pos)),{}).values())
    def distribution(self,words,index,lemma,pos):
        rows=self.model.rank_components(self.model.components(words,index,lemma,pos))
        if not rows:return {}
        return dict(zip([r['sense'] for r in rows],softmax([r['score'] for r in rows],self.temperature)))

class GlossSense(SenseDistribution):
    """Unsupervised: cosine between averaged context vectors and each sense's WordNet gloss."""
    name='gloss'
    def __init__(self,temperature=.05):self.temperature=temperature
    def distribution(self,words,index,lemma,pos):
        senses=synsets(lemma,word_pos(pos))
        if not senses:return {}
        v=context_embedding(words,index)
        return dict(zip([s.name() for s in senses],softmax([float(v@gloss(s.name())) for s in senses],self.temperature)))

class SenseEnsemble(SenseDistribution):
    """Weighted mixture of member distributions. Weights are chosen on SemCor development data.

    With evidence>0, the first member (SemCor) keeps weight w*n/(n+evidence) for a lemma with n training
    examples; the rest moves to the other members. One SemCor example no longer decides the sense.
    """
    name='sense_ensemble'
    def __init__(self,members,weights,evidence=0.):self.members,self.weights,self.evidence=members,list(weights),evidence
    def member_weights(self,lemma,pos):
        w=list(self.weights)
        if self.evidence and hasattr(self.members[0],'examples'):
            n=self.members[0].examples(lemma,pos);keep=w[0]*n/(n+self.evidence)
            rest=sum(w[1:]);k=len(w)-1
            # Released weight follows the other members' weights, or splits evenly if they are all zero.
            w=[keep]+[(x/rest if rest else 1/k)*(1-keep) for x in w[1:]]
        return w
    def distribution(self,words,index,lemma,pos):
        out={}
        for m,w in zip(self.members,self.member_weights(lemma,pos)):
            if not w:continue
            for s,p in m.distribution(words,index,lemma,pos).items():out[s]=out.get(s,0.)+w*p
        z=sum(out.values())
        return {s:p/z for s,p in out.items()} if z else {}

# Fit: (slot, candidates) -> one score per candidate ------------------------------------------

class FitScorer(ABC):
    name='fit'
    @abstractmethod
    def score(self,slot,candidates):
        """Higher means the candidate better preserves the target's meaning in this context."""

def target_senses(slot,sense_model):
    if 'senses' not in slot.cache:slot.cache['senses']=sense_model.distribution(slot.words,slot.index,slot.lemma,slot.pos)
    return slot.cache['senses']

class SenseFit(FitScorer):
    """Probability that the target's intended sense is one the candidate can express."""
    name='sense_fit'
    def __init__(self,sense_model):self.sense_model=sense_model
    def score(self,slot,candidates):
        p=target_senses(slot,self.sense_model)
        return np.array([sum(p.get(s,0.) for s in c['senses']) for c in candidates])

class FirstSenseFit(FitScorer):
    """WordNet sense order: 1 when the candidate expresses the target's first-listed sense, 1/2 for the second, ...

    WordNet orders senses by tagged-corpus frequency, largely SemCor. It is therefore a stacker member trained on
    SWORDS, never part of the SemCor-tuned sense mixture, so SemCor test scores stay free of that overlap.
    """
    name='first_sense'
    def score(self,slot,candidates):
        order={s.name():i for i,s in enumerate(synsets(slot.lemma,word_pos(slot.pos)))}
        return np.array([1/(1+min((order[s] for s in c['senses'] if s in order),default=np.inf)) for c in candidates])

class CandidateSenseRank(FitScorer):
    """How central the intended sense is for the candidate itself: 1/(1+position) of the target's most
    probable shared sense in the candidate's own WordNet sense list. qualify lists the stipulate sense late,
    so "stipulates -> qualifies" scores low even though WordNet groups the two words together.
    """
    name='candidate_sense_rank'
    def __init__(self,sense_model):self.sense_model=sense_model
    def score(self,slot,candidates):
        p=target_senses(slot,self.sense_model);out=[]
        for c in candidates:
            shared=[s for s in c['senses'] if s in p]
            if not shared:out.append(0.);continue
            best=max(shared,key=lambda s:p[s])
            own=[x.name() for x in synsets(c['lemma'].lower().replace(' ','_'),word_pos(slot.pos))]
            out.append(1/(1+own.index(best)) if best in own else 0.)
        return np.array(out)

class RoundTripFit(FitScorer):
    """Probability that a reader of the rewritten sentence infers the same sense.

    Catches candidates that share a rare sense with the target but usually mean something else,
    such as exasperate for exacerbate, by disambiguating the candidate in the substituted sentence.
    """
    name='round_trip'
    def __init__(self,sense_model):self.sense_model=sense_model
    def score(self,slot,candidates):
        p=target_senses(slot,self.sense_model);out=[]
        for c in candidates:
            q=self.sense_model.distribution(slot.substituted(c),slot.index,c['lemma'].lower().replace(' ','_'),slot.pos)
            out.append(sum(p.get(s,0.)*q.get(s,0.) for s in set(p)&set(q)))
        return np.array(out)

def vector_rows(slot,candidates):
    key=tuple(c['word'] for c in candidates)
    if slot.cache.get('vector_key')!=key:
        if 'gloss_senses' not in slot.cache:slot.cache['gloss_senses']=sense_scores(slot.context_doc,slot.context_token)
        slot.cache['vector_rows']=[vector_features(slot.context_doc,slot.context_token,c,senses=slot.cache['gloss_senses']) for c in candidates]
        slot.cache['vector_key']=key
    return slot.cache['vector_rows']

def checker_model():
    """The SWORDS context checker's estimator (settings from train_context_v2.py; round C found no better ones)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.ensemble import HistGradientBoostingClassifier
    return make_pipeline(DictVectorizer(sparse=False),StandardScaler(),
                         HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=10,random_state=4701))

class CheckerFit(FitScorer):
    """The SWORDS-trained v2 context checker, used as a soft score rather than a hard threshold."""
    name='swords_checker'
    def __init__(self,checker):self.checker=checker
    def score(self,slot,candidates):
        rows=vector_rows(slot,candidates)
        return self.checker.predict_proba(rows)[:,1] if rows else np.array([])

class ContextVectorFit(FitScorer):
    """Change in similarity to the whole sentence when the candidate replaces the target."""
    name='context_vector'
    def score(self,slot,candidates):
        return np.array([r['vector_context_delta'] for r in vector_rows(slot,candidates)])

ARGUMENTS={'nsubj','nsubjpass','dobj','obj','pobj','amod','advmod','compound','attr','acomp','prep'}

class ArgumentFit(FitScorer):
    """Change in vector similarity to syntactically linked words, e.g. relieve/facilitate versus pain."""
    name='argument_fit'
    def score(self,slot,candidates):
        t=slot.context_token
        linked=[c for c in t.children if c.dep_ in ARGUMENTS and c.is_alpha]
        if t.dep_ in ARGUMENTS and t.head.is_alpha:linked.append(t.head)
        linked=[x for x in linked if not x.is_stop]
        if not linked:return np.zeros(len(candidates))
        vs=[embed(x.lemma_) for x in linked];source=embed(t.lemma_)
        base=np.mean([source@v for v in vs])
        return np.array([np.mean([embed(c['lemma'])@v for v in vs])-base for c in candidates])

class LanguageModelFit(FitScorer):
    """Brown bigram log-probability change around the target."""
    name='bigram'
    def score(self,slot,candidates):
        t=slot.token;old=local_lm(slot.doc,t,t.text)
        return np.array([local_lm(slot.doc,t,c['word'])-old for c in candidates])

class WordNetMember(FitScorer):
    """Constant 1 for generated candidates; lets the stacker train on SWORDS pairs outside WordNet."""
    name='wordnet_member'
    def score(self,slot,candidates):return np.array([float(bool(c['senses'])) for c in candidates])

class CandidateFlag(FitScorer):
    """1 when a generator attribute matches, e.g. source='hypernym' or multiword=True. Lets the stacker learn
    how risky each generation route is; hypernyms can lose specificity that no guardrail detects."""
    def __init__(self,key,value=True):self.key,self.value=key,value;self.name=f'{key}={value}'
    def score(self,slot,candidates):return np.array([float(c.get(self.key)==self.value) for c in candidates])

class CandidateFrequency(FitScorer):
    """Not a meaning signal; lets the stacker learn that very rare candidates are rarely accepted."""
    name='candidate_zipf'
    def score(self,slot,candidates):
        return np.array([base_features(c['word'])['zipf']-base_features(slot.token.text)['zipf'] for c in candidates])

class StackedFit(FitScorer):
    """Logistic regression over member scores; outputs a probability that the substitute fits.

    fit() uses labeled (slot, candidate) pairs. Members are frozen; only the combiner is trained.
    """
    name='stacked_fit'
    def __init__(self,members,C=1.):
        self.members=members;self.C=C;self.scaler=None;self.model=None
    def features(self,slot,candidates):
        if not candidates:return np.zeros((0,len(self.members)))
        return np.column_stack([m.score(slot,candidates) for m in self.members])
    def fit(self,X,y):
        self.scaler=StandardScaler().fit(X)
        self.model=LogisticRegression(C=self.C,max_iter=2000).fit(self.scaler.transform(X),y)
        return self
    def proba(self,X):
        if len(X)==0:return np.array([])
        return self.model.predict_proba(self.scaler.transform(X))[:,1]
    def score(self,slot,candidates):return self.proba(self.features(slot,candidates))
    def weights(self):
        return {m.name:float(w) for m,w in zip(self.members,self.model.coef_[0])}

class AverageFit(FitScorer):
    """Unweighted mean of member probabilities; an untrained comparison for StackedFit."""
    name='average_fit'
    def __init__(self,members):self.members=members
    def score(self,slot,candidates):
        return np.mean([m.score(slot,candidates) for m in self.members],axis=0) if candidates else np.array([])

# Decision: (target, scored candidates) -> probability each candidate is a good simplification --------

class DecisionModel(ABC):
    name='decision'
    @abstractmethod
    def probabilities(self,case):
        """case: {'target': word, 'candidates': [rows with fit, gains, member scores, source, guard]}."""

class LearnedDecision(DecisionModel):
    """A classifier over per-candidate features built by featurizer(case) -> f(case, candidate)."""
    name='learned_decision'
    def __init__(self,model,featurizer):self.model,self.featurizer=model,featurizer
    def probabilities(self,case):
        if not case['candidates']:return np.array([])
        f=self.featurizer(case)
        return self.model.predict_proba(np.array([f(case,c) for c in case['candidates']]))[:,1]
