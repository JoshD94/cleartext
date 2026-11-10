"""Which word gets the single edit? Rules scored on two dev signals at once:
BenchLS (all 929 as dev): edit lands on the annotated target with a gold replacement.
CWI dev: edit lands on a word most of 20 annotators marked hard, minus edits on words most did not.
TSAR test and CWI test are report-only.
"""
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cleartext.data import ROOT
from cleartext.ensemble_pipeline import LATEST
import os
TV,CV=os.environ.get('TARGET_CACHE','v5'),os.environ.get('CWI_CACHE','v1')
C=lambda n:pickle.load(open(ROOT/f'data/cache/{n}.pkl','rb'))
bench=C(f'refine-targets-{TV}-benchls-dev')+C(f'refine-targets-{TV}-benchls-holdout');tsar=C(f'refine-targets-{TV}-tsar-test')
cwi_dev,cwi_test=C(f'refine-cwi-targets-{CV}-Dev'),C(f'refine-cwi-targets-{CV}-Test')
def pick(case,alpha,detect_t,noun_t):
    ok=[w for w in case['words'] if w['replacement'] and w['detect']>=detect_t and (w['pos']!='NOUN' or w['detect']>=noun_t)]
    # alpha 0: decision probability only; large alpha: detector dominates.
    return max(ok,key=lambda w:w['accept']*w['detect']**alpha) if ok else None
def bench_score(cases,rule):
    chosen=[pick(c,*rule) for c in cases];return sum(bool(w and w['is_target'] and w['gold']) for w in chosen),sum(w is not None for w in chosen)
def cwi_score(cases,rule):
    chosen=[w for w in (pick(c,*rule) for c in cases) if w and w['hardness'] is not None]
    hard=sum(w['hardness']>=.5 for w in chosen);return 2*hard-len(chosen),hard,len(chosen)
rules=[(a,d,n) for a in [0,.5,1,2,4,8] for d in [.39,.5,.6] for n in [0,.6,.7,.8]]
rows=[]
for r in rules:
    b,be=bench_score(bench,r);cn,ch,ce=cwi_score(cwi_dev,r);rows.append({'rule':r,'bench':b,'bench_edits':be,'cwi_net':cn,'cwi_hard':ch,'cwi_edits':ce})
cur=next(x for x in rows if x['rule']==(0,.39,0))
# Keep rules that do not lose BenchLS correct target edits versus the current rule, then take the best CWI net.
eligible=[x for x in rows if x['bench']>=cur['bench']] 
best=max(eligible,key=lambda x:(x['cwi_net'],x['bench']))
front=sorted(rows,key=lambda x:(-x['bench'],-x['cwi_net']))
print('current',cur);print('chosen ',best)
print('top BenchLS rules:',[(x['rule'],x['bench'],x['cwi_net']) for x in front[:6]])
print('top CWI rules    :',[(x['rule'],x['bench'],x['cwi_net']) for x in sorted(rows,key=lambda x:-x['cwi_net'])[:6]])
for label,r in [('current',cur['rule']),('chosen',best['rule'])]:
    tb,te=bench_score(tsar,r);cn,ch,ce=cwi_score(cwi_test,r)
    print(f"{label:<8} {r}: TSAR correct on target {tb} of {te} edits | CWI test {ch}/{ce} edits on hard words ({ch/ce:.2f}), net {cn}")
json.dump({'rows':[{**x,'rule':list(x['rule'])} for x in rows],'current':cur['rule'],'chosen':list(best['rule'])},open(LATEST/'refine_target_joint.json','w'),indent=2)
