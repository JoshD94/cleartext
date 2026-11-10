"""Shared helpers for decision experiments on cached candidate tables (scripts/build_candidate_tables.py)."""
import pickle,re
from functools import lru_cache
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from wordfreq import zipf_frequency
from .data import ROOT
from .features import base_features

SOURCES=['synonym','hypernym','similar']
IN_SENTENCE=False  # round 5 feature; tested, no dev gain, so off

def tables(version='v2'):
    load=lambda s:pickle.load(open(ROOT/f'data/cache/refine-{version}-{s}.pkl','rb'))
    names,dev=load('benchls-dev')
    return names,{'benchls_dev':dev,'benchls_holdout':load('benchls-holdout')[1],'tsar_test':load('tsar-test')[1]}

def evaluate(cases,choose):
    edits=correct=0
    for case in cases:
        c=choose(case)
        if c is not None:edits+=1;correct+=c['gold']
    return {'edits':edits,'correct':correct,'wrong':edits-correct,'net':2*correct-edits,
            'precision':round(correct/edits,3) if edits else 0.,'n':len(cases)}

@lru_cache(100000)
def zipf(w):return zipf_frequency(w.lower(),'en')

def features(case,c):
    """Member scores + SWORDS fit + both CompLex gains + source flags + raw frequency/length/syllable differences."""
    t=case['target'];bt,bc=base_features(t),base_features(c['word'].split()[0])
    return np.r_[c['x'],c['fit'],c['gain_word'],c['gain_context'],c['target_difficulty'],
                 [float(c['source']==s) for s in SOURCES],float(c['multiword']),
                 zipf(c['word'])-zipf(t),zipf(t),bc['length']-bt['length'],bc['syllables']-bt['syllables']]

def feature_names(names):
    return names+['fit','gain_word','gain_context','target_difficulty']+[f'source={s}' for s in SOURCES]+['multiword',
                  'zipf_diff','target_zipf','length_diff','syllable_diff']

def flat(cases,featurize=features):
    X,y,groups,keys=[],[],[],[]
    for case in cases:
        for c in case['candidates']:
            X.append(featurize(case,c));y.append(c['gold']);groups.append(case['target'].lower());keys.append((case['id'],c['word']))
    return np.array(X),np.array(y,bool),np.array(groups),keys

def accept_model(C=.1):return make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000))

def oof_probs(X,y,groups,C=.1,model=accept_model,n_splits=5):
    """Out-of-fold probabilities, folds grouped by target word."""
    p=np.zeros(len(y))
    for a,b in GroupKFold(n_splits=n_splits).split(X,y,groups):p[b]=model(C).fit(X[a],y[a]).predict_proba(X[b])[:,1]
    return p

def accept_rule(prob,threshold,guard=True):
    """Pick the most probable candidate at or above threshold that passes the guardrails."""
    def choose(case):
        ok=[c for c in case['candidates'] if (c['guard'] or not guard) and prob[case['id'],c['word']]>=threshold]
        return max(ok,key=lambda c:prob[case['id'],c['word']]) if ok else None
    return choose

def best_threshold(cases,prob,grid=np.round(np.arange(.1,.8,.05),2)):
    scores={t:evaluate(cases,accept_rule(prob,t))['net'] for t in grid}
    t=max(scores,key=scores.get);return float(t),scores[t]

def spelling_variant(a,b):
    """organisation/organization, colour/color: the same word, not a simplification."""
    n=lambda w:w.lower().replace('isa','iza').replace('ise','ize').replace('yse','yze').replace('our','or').replace('tre','ter')
    return a.lower()!=b.lower() and n(a)==n(b)

def shared_prefix(a,b):
    a,b=a.lower(),b.lower();n=0
    while n<min(len(a),len(b)) and a[n]==b[n]:n+=1
    return n/max(len(a),1)

def features_listwise(case):
    """Per-case feature function: adds each candidate's standing among the other candidates for the same target."""
    cands=case['candidates']
    if not cands:return lambda case,c:None
    fits=np.array([c['fit'] for c in cands]);zs=np.array([zipf(c['word']) for c in cands])
    order={c['word']:r for r,c in enumerate(sorted(cands,key=lambda c:-c['fit']))}
    others=set(re.findall(r'[a-z]+',case.get('text','').lower()))-{case['target'].lower()}
    def f(case,c):
        row=np.r_[features(case,c),c['fit']-fits.max(),zipf(c['word'])-zs.max(),order[c['word']],len(cands),
                  shared_prefix(c['word'],case['target']),float(spelling_variant(c['word'],case['target']))]
        # Round 5 feature: the candidate (or its lemma) already appears elsewhere in the sentence.
        return np.r_[row,float(bool({c['word'].lower(),c['lemma'].lower()}&others))] if IN_SENTENCE else row
    return f

def flat_listwise(cases):
    X,y,groups,keys=[],[],[],[]
    for case in cases:
        f=features_listwise(case)
        for c in case['candidates']:
            X.append(f(case,c));y.append(c['gold']);groups.append(case['target'].lower());keys.append((case['id'],c['word']))
    return np.array(X),np.array(y,bool),np.array(groups),keys

# ---- Round 6: how much more general is the candidate's sense than the target's? ----
from nltk.corpus import wordnet as wn
SPECIFICITY=True
POS_FEATURES=True  # round 10

@lru_cache(50000)
def descendants(name,cap=20000):
    """Number of senses filed under a WordNet sense (hyponym closure), capped for the very top nodes."""
    s=wn.synset(name);seen=set();stack=[s]
    while stack and len(seen)<cap:
        for h in stack.pop().hyponyms()+[]:
            if h.name() not in seen:seen.add(h.name());stack.append(h)
    return len(seen)

@lru_cache(200000)
def specificity(target_lemma,pos,cand_lemma):
    """(specificity loss, candidate depth). Loss = log descendants of the candidate's sense minus the target's,
    minimized over the target senses the candidate was reached from. Synonyms share the sense, so loss is 0."""
    from .lexical import synsets
    from .wsd import word_pos
    key=cand_lemma.lower().replace(' ','_');best=None
    for s in synsets(target_lemma,word_pos(pos)):
        linked=[s]+s.hypernyms()+s.verb_groups()+s.similar_tos()+s.also_sees()
        for h in linked:
            if key in {l.lower() for l in h.lemma_names()}:
                loss=np.log1p(descendants(h.name()))-np.log1p(descendants(s.name()))
                row=(float(loss),float(h.min_depth()))
                if best is None or row[0]<best[0]:best=row
    return best or (0.,0.)

def features_specific(case):
    base=features_listwise(case);info=case.get('target_info')
    def f(case,c):
        row=base(case,c)
        if not SPECIFICITY:return row
        loss,depth=specificity(info[0],info[1],c['lemma']) if info else (0.,0.)
        row=np.r_[row,loss,depth]
        # Round 10: target part of speech (WordNet n/v/a/r).
        return np.r_[row,[float((info or (None,'x'))[1]==p) for p in 'nvar']] if POS_FEATURES else row
    return f

def flat_specific(cases):
    X,y,groups,keys=[],[],[],[]
    for case in cases:
        f=features_specific(case)
        for c in case['candidates']:
            X.append(f(case,c));y.append(c['gold']);groups.append(case['target'].lower());keys.append((case['id'],c['word']))
    return np.array(X),np.array(y,bool),np.array(groups),keys

class AverageDecision:
    """Mean probability of a logistic model and a small gradient-boosting model (round 7). sklearn-style fit/predict_proba."""
    def __init__(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.members=[accept_model(.03),HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=4,
                      l2_regularization=1.,min_samples_leaf=20,random_state=4701)]
    def fit(self,X,y):
        for m in self.members:m.fit(X,y)
        return self
    def predict_proba(self,X):return np.mean([m.predict_proba(X) for m in self.members],axis=0)

# ---- Round 8: signals from existing blocks (scripts/experiments/refine_extras.py) ----
EXTRA_GROUPS={'detector':['det_target','det_cand','det_gain'],'sentence':['sent_delta'],'antonym':['antonym']}

def load_extras():
    out={}
    for s in ['benchls-dev','benchls-holdout','tsar-test']:out.update(pickle.load(open(ROOT/f'data/cache/refine-extras-v1-{s}.pkl','rb')))
    return out

def extra_values(extras,case,c,groups):
    e=extras.get((case['id'],c['word']),{'det_target':0.,'det_cand':0.,'sent_delta':0.,'antonym':0.})
    e={**e,'det_gain':e['det_target']-e['det_cand']}
    return [e[k] for g in groups for k in EXTRA_GROUPS[g]]

def cv_decision(X,y,g,keys,cases,model,repeats=3,folds=10):
    """Round-7 protocol: repeated grouped k-fold; returns mean net at the best threshold, log-loss, median threshold."""
    from sklearn.metrics import log_loss
    nets,losses,ts=[],[],[]
    for seed in range(repeats):
        rng=np.random.default_rng(seed);ug=np.unique(g);perm=dict(zip(ug,rng.permutation(len(ug))%folds))
        fold=np.array([perm[x] for x in g]);p=np.zeros(len(y))
        for f in range(folds):
            tr,te=fold!=f,fold==f;p[te]=model().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
        t,net=best_threshold(cases,dict(zip(keys,p)));nets.append(net);ts.append(t);losses.append(log_loss(y,np.clip(p,1e-6,1-1e-6)))
    return float(np.mean(nets)),float(np.mean(losses)),float(np.median(ts)),nets

class WeightedDecision(AverageDecision):
    """AverageDecision with tunable members and mixing weight (round 12)."""
    def __init__(self,C=.03,leaves=4,weight=.5):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.weight=weight
        self.members=[accept_model(C),HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=leaves,
                      l2_regularization=1.,min_samples_leaf=20,random_state=4701)]
    def predict_proba(self,X):
        return self.weight*self.members[0].predict_proba(X)+(1-self.weight)*self.members[1].predict_proba(X)
