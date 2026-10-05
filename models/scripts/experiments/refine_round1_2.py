"""Round-based decision experiments on cached candidate tables.

Every variant is chosen on BenchLS dev (out-of-fold when trained there). BenchLS holdout is fresh; TSAR test
is reused. Both are reported only for the dev-selected variants.
"""
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from cleartext.data import ROOT
from cleartext.ensemble_pipeline import utility
names,dev=pickle.load(open(ROOT/'data/cache/refine-v2-benchls-dev.pkl','rb'))
_,hold=pickle.load(open(ROOT/'data/cache/refine-v2-benchls-holdout.pkl','rb'))
_,tsar=pickle.load(open(ROOT/'data/cache/refine-v2-tsar-test.pkl','rb'))
SETS={'benchls_dev':dev,'benchls_holdout':hold,'tsar_test':tsar}

def gain(c,kind):return {'word':c['gain_word'],'context':c['gain_context'],'mean':(c['gain_word']+c['gain_context'])/2}[kind]

def evaluate(cases,choose):
    edits=correct=0
    for case in cases:
        c=choose(case)
        if c is not None:edits+=1;correct+=c['gold']
    return {'edits':edits,'correct':correct,'wrong':edits-correct,'net':2*correct-edits,
            'precision':round(correct/edits,3) if edits else 0.,'n':len(cases),
            'reachable':sum(any(c['gold'] for c in case['candidates']) for case in cases)}

def utility_rule(cost,tiered=True,g='word'):
    def choose(case):
        ok=[c for c in case['candidates'] if c['guard'] and gain(c,g)>0 and utility(c['fit'],gain(c,g),cost)>0]
        if not ok:return None
        return max(ok,key=lambda c:(-int(tiered and c['source']!='synonym'),utility(c['fit'],gain(c,g),cost)))
    return choose

SOURCES=['synonym','hypernym','similar']
def features(c):
    return np.r_[c['x'],c['fit'],c['gain_word'],c['gain_context'],c['target_difficulty'],
                 [float(c['source']==s) for s in SOURCES],float(c['multiword'])]

def accept_model(C=.3):
    return make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000))

def accept_rule(prob,threshold,tiered=False,g='word'):
    """prob: {(case id, word): probability the candidate is a gold simplification}."""
    def choose(case):
        ok=[c for c in case['candidates'] if c['guard'] and gain(c,g)>0 and prob[case['id'],c['word']]>=threshold]
        if not ok:return None
        return max(ok,key=lambda c:(-int(tiered and c['source']!='synonym'),prob[case['id'],c['word']]))
    return choose

def flat(cases):
    X,y,groups,keys=[],[],[],[]
    for case in cases:
        for c in case['candidates']:
            X.append(features(c));y.append(c['gold']);groups.append(case['target'].lower());keys.append((case['id'],c['word']))
    return np.array(X),np.array(y),np.array(groups),keys

results={}
def report(label,rule_for):
    """rule_for(split) -> choose function. Records all three sets; selection reads only benchls_dev."""
    results[label]={s:evaluate(cases,rule_for(s)) for s,cases in SETS.items()}
    r=results[label];print(f"{label:<44} dev net {r['benchls_dev']['net']:>4} ({r['benchls_dev']['correct']}/{r['benchls_dev']['edits']})"
                           f"  | holdout {r['benchls_holdout']['correct']}/{r['benchls_holdout']['edits']}"
                           f"  | tsar {r['tsar_test']['correct']}/{r['tsar_test']['edits']}",flush=True)

# Round 1a: the shipped v2 decision (SWORDS-tuned cost 0.05, tiered).
report('v2 as shipped (cost .05, tiered)',lambda s:utility_rule(.05))
# Round 1b: same rule, cost, tiering and gain model retuned on BenchLS dev.
grid=[(cost,tiered,g) for cost in [0,.01,.02,.03,.05,.075,.1,.15,.2] for tiered in [True,False] for g in ['word','context','mean']]
dev_scores={k:evaluate(dev,utility_rule(*k))['net'] for k in grid}
best=max(grid,key=lambda k:dev_scores[k])
report(f'utility retuned on BenchLS dev {best}',lambda s:utility_rule(*best))
best_gain={g:max((k for k in grid if k[2]==g),key=lambda k:dev_scores[k]) for g in ['word','context','mean']}
for g,k in best_gain.items():print('   best with gain',g,k,'dev net',dev_scores[k])

# Round 1c: a decision model trained on the target task (BenchLS dev), stacked on the SWORDS-trained fit.
X,y,groups,keys=flat(dev)
folds=list(GroupKFold(n_splits=5).split(X,y,groups))
oof={}
for C in [.03,.1,.3,1.]:
    p=np.zeros(len(y))
    for a,b in folds:p[b]=accept_model(C).fit(X[a],y[a]).predict_proba(X[b])[:,1]
    oof[C]=dict(zip(keys,p))
choices=[(C,t,tiered,g) for C in oof for t in np.round(np.arange(.1,.8,.05),2) for tiered in [False,True] for g in ['word','mean']]
acc_scores={k:evaluate(dev,accept_rule(oof[k[0]],k[1],k[2],k[3]))['net'] for k in choices}
abest=max(choices,key=lambda k:acc_scores[k])
final=accept_model(abest[0]).fit(X,y)
def probs(cases):
    Xs,_,_,ks=flat(cases);return dict(zip(ks,final.predict_proba(Xs)[:,1])) if len(ks) else {}
P={'benchls_dev':oof[abest[0]],'benchls_holdout':probs(hold),'tsar_test':probs(tsar)}
report(f'accept model on BenchLS dev {abest}',lambda s:accept_rule(P[s],*abest[1:]))
coef=dict(zip(names+['fit','gain_word','gain_context','target_difficulty']+[f'source={s}' for s in SOURCES]+['multiword'],final[-1].coef_[0]))
print('   accept model weights',json.dumps({k:round(float(v),2) for k,v in sorted(coef.items(),key=lambda x:-abs(x[1]))}))
json.dump({'results':results,'utility_best':best,'accept_best':[float(abest[0]),float(abest[1]),abest[2],abest[3]],'accept_weights':{k:float(v) for k,v in coef.items()}},
          open(ROOT/'runs/ensemble-v2-20260927/refine_round1.json','w'),indent=2)

# ---------------- Round 2 ----------------
# The CompLex difficulty model calls many human-chosen substitutes "not simpler", and the hard gain>0 gate drops
# them. Here the decision model sees gain as evidence instead of a gate, plus raw frequency and length differences
# so BenchLS labels can learn "simpler" directly. Guard variants: all checks, or all but argument_roles.
from functools import lru_cache
from wordfreq import zipf_frequency
from cleartext.features import base_features
@lru_cache(100000)
def zipf(w):return zipf_frequency(w.lower(),'en')
def features2(case,c):
    t=case['target'];bt,bc=base_features(t),base_features(c['word'].split()[0])
    return np.r_[features(c),zipf(c['word'])-zipf(t),zipf(t),bc['length']-bt['length'],bc['syllables']-bt['syllables']]
def guard_ok(c,mode):
    return c['guard'] if mode=='all' else not [f for f in c['guard_failed'] if f!='argument_roles']
def flat2(cases):
    X,y,groups,keys=[],[],[],[]
    for case in cases:
        for c in case['candidates']:
            X.append(features2(case,c));y.append(c['gold']);groups.append(case['target'].lower());keys.append((case['id'],c['word']))
    return np.array(X),np.array(y),np.array(groups),keys
def accept_rule2(prob,threshold,gate,mode):
    def choose(case):
        ok=[c for c in case['candidates'] if guard_ok(c,mode) and (not gate or c['gain_word']>0) and prob[case['id'],c['word']]>=threshold]
        return max(ok,key=lambda c:prob[case['id'],c['word']]) if ok else None
    return choose
X2,y2,g2,k2=flat2(dev);folds2=list(GroupKFold(n_splits=5).split(X2,y2,g2))
for feats,cols in [('round-1 features',slice(0,X.shape[1])),('+ frequency/length',slice(None))]:
    oof2={}
    for C in [.03,.1,.3,1.]:
        p=np.zeros(len(y2))
        for a,b in folds2:p[b]=accept_model(C).fit(X2[a][:,cols],y2[a]).predict_proba(X2[b][:,cols])[:,1]
        oof2[C]=dict(zip(k2,p))
    choices=[(C,t,gate,mode) for C in oof2 for t in np.round(np.arange(.1,.8,.05),2) for gate in [True,False] for mode in ['all','no_roles']]
    sc={k:evaluate(dev,accept_rule2(oof2[k[0]],*k[1:]))['net'] for k in choices}
    for gate in [True,False]:
        for mode in ['all','no_roles']:
            k=max((c for c in choices if c[2]==gate and c[3]==mode),key=lambda c:sc[c])
            print(f'   {feats:<20} gate={gate!s:<5} guard={mode:<8} dev net {sc[k]:>4}  (C={k[0]}, t={k[1]})')
    k=max(choices,key=lambda c:sc[c])
    m=accept_model(k[0]).fit(X2[:,cols],y2)
    def pr(cases):
        Xs,_,_,ks=flat2(cases);return dict(zip(ks,m.predict_proba(Xs[:,cols])[:,1]))
    P2={'benchls_dev':oof2[k[0]],'benchls_holdout':pr(hold),'tsar_test':pr(tsar)}
    report(f'accept, {feats} {k}',lambda s:accept_rule2(P2[s],*k[1:]))
json.dump({'results':results},open(ROOT/'runs/ensemble-v2-20260927/refine_round2.json','w'),indent=2,default=str)
