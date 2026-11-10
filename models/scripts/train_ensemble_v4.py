"""Ensemble v4: v3 with the round-7 decision model (average of logistic and boosting, plus WordNet specificity).

Chosen on all of BenchLS as dev (10-fold x3 grouped, net and log-loss agree). TSAR test is report-only.
"""
import sys,json,pickle,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cleartext.data import ROOT,tsar_data
from cleartext.features import parse,target_token
import cleartext.refine as R
from cleartext.ensemble_pipeline import EnsembleClearText
V3=ROOT/'runs/ensemble-v3-20260927';V4=ROOT/'runs/ensemble-v4-20260927';V4.mkdir(exist_ok=True)
THRESHOLD=.45
names,S=R.tables();R.SPECIFICITY=True
X,y,_,_=R.flat_specific(S['benchls_dev']+S['benchls_holdout']);model=R.AverageDecision().fit(X,y)
pickle.dump(model,open(V4/'decision.pkl','wb'))
for f in ['stacker.pkl','checker.pkl','detector.pkl']:shutil.copy(V3/f,V4/f)
config=json.loads((V3/'config.json').read_text())
config.update(checker=str((V4/'checker.pkl').relative_to(ROOT)),detector=str((V4/'detector.pkl').relative_to(ROOT)),
              decision=str((V4/'decision.pkl').relative_to(ROOT)),decision_features='specific',decision_threshold=THRESHOLD)
(V4/'config.json').write_text(json.dumps(config,indent=2))
Xt,_,_,kt=R.flat_specific(S['tsar_test'])
table=R.evaluate(S['tsar_test'],R.accept_rule(dict(zip(kt,model.predict_proba(Xt)[:,1])),THRESHOLD))
pipe=EnsembleClearText.load(V4)
rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test');e=c=0;out=[]
for r,d in zip(rows,docs):
    sel=pipe.rank_word(d,target_token(d,r['target']))['selected']
    if sel:e+=1;c+=sel['word'].lower() in r['gold']
    out.append({'target':r['target'],'replacement':sel['word'] if sel else None,'correct':bool(sel and sel['word'].lower() in r['gold'])})
print('TABLE TSAR',json.dumps({k:table[k] for k in ['edits','correct','precision','net']}))
print('PIPELINE TSAR',json.dumps({'edits':e,'correct':c,'precision':round(c/e,3),'net':2*c-e}))
examples=[r['original'] for r in json.loads((ROOT/'runs/initial-20260924/basic_examples_20.json').read_text())]
examples+=['The medication alleviates pain but can cause drowsiness.','Heavy rainfall may exacerbate the flooding.',
           'She deposited the money at the bank.','The gene encodes a protein that regulates growth.','It was an auspicious start to the season.']
samples=[pipe.analyze(s,structure=False,phrases=False) for s in examples]
for s in samples:print('SAMPLE',('* ' if s['edits'] else '  ')+s['output'],flush=True)
json.dump({'tsar_table':table,'tsar_pipeline':{'edits':e,'correct':c},'tsar_outputs':out,
           'samples':[{'original':s['original'],'output':s['output']} for s in samples],'threshold':THRESHOLD},
          open(V4/'evaluation.json','w'),indent=2,default=float)
