"""Ensemble v3: v2 plus a decision model trained on BenchLS (the lexical simplification task itself).

Settings (logistic, C=0.03, threshold 0.40, listwise features, no hard gain gate) were chosen on BenchLS dev
in refine_round1-4. decision-dev.pkl (dev only) gives the honest holdout and TSAR numbers; decision.pkl is
refit on all of BenchLS after the holdout was reported and is what the pipeline ships.
"""
import sys,json,pickle,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cleartext.data import ROOT,tsar_data
from cleartext.features import parse,target_token
from cleartext.refine import tables,flat_listwise,accept_model,accept_rule,evaluate
from cleartext.ensemble_pipeline import EnsembleClearText,RUN
V3=ROOT/'runs/ensemble-v3-20260927';V3.mkdir(exist_ok=True)
C,THRESHOLD=.03,.40
names,S=tables()
X,y,_,_=flat_listwise(S['benchls_dev']);dev_model=accept_model(C).fit(X,y)
Xa,ya,_,_=flat_listwise(S['benchls_dev']+S['benchls_holdout']);full_model=accept_model(C).fit(Xa,ya)
pickle.dump(dev_model,open(V3/'decision-dev.pkl','wb'));pickle.dump(full_model,open(V3/'decision.pkl','wb'))
for f in ['stacker.pkl','checker.pkl']:shutil.copy(RUN/f,V3/f)
config=json.loads((RUN/'config.json').read_text())
config.update(checker=str((V3/'checker.pkl').relative_to(ROOT)),decision=str((V3/'decision-dev.pkl').relative_to(ROOT)),
              decision_threshold=THRESHOLD,tiered=False)
(V3/'config.json').write_text(json.dumps(config,indent=2))

def run_tsar(pipe):
    rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test');out=[]
    for r,d in zip(rows,docs):
        sel=pipe.rank_word(d,target_token(d,r['target']))['selected']
        out.append({'id':r['id'],'target':r['target'],'replacement':sel['word'] if sel else None,'changed':sel is not None,
                    'correct':bool(sel and sel['word'].lower() in r['gold']),'source':sel['source'] if sel else None})
    e=sum(o['changed'] for o in out);c=sum(o['correct'] for o in out)
    return {'edits':e,'correct':c,'precision':round(c/e,3),'net':2*c-e},out
# Parity: the end-to-end pipeline with the dev-only model must reproduce the table evaluation.
table={s:evaluate(S[s],accept_rule(dict(zip(flat_listwise(S[s])[3],dev_model.predict_proba(flat_listwise(S[s])[0])[:,1])),THRESHOLD)) for s in S}
dev_pipe,dev_out=run_tsar(EnsembleClearText.load(V3))
print('TABLE (dev-only model)',json.dumps({s:{k:table[s][k] for k in ['edits','correct','precision','net']} for s in S}))
print('PIPELINE TSAR (dev-only model)',json.dumps(dev_pipe))
config['decision']=str((V3/'decision.pkl').relative_to(ROOT));(V3/'config.json').write_text(json.dumps(config,indent=2))
full_pipe,full_out=run_tsar(EnsembleClearText.load(V3))
print('PIPELINE TSAR (shipped, all BenchLS)',json.dumps(full_pipe))
pipe=EnsembleClearText.load(V3)
examples=[r['original'] for r in json.loads((ROOT/'runs/initial-20260924/basic_examples_20.json').read_text())]
examples+=['The medication alleviates pain but can cause drowsiness.','Heavy rainfall may exacerbate the flooding.',
           'She deposited the money at the bank.','The gene encodes a protein that regulates growth.','It was an auspicious start to the season.']
samples=[pipe.analyze(s,structure=False,phrases=False) for s in examples]
for s in samples:print('SAMPLE',s['original'],'=>',s['output'],flush=True)
json.dump({'table_dev_only':table,'tsar_pipeline_dev_only':dev_pipe,'tsar_pipeline_shipped':full_pipe,'tsar_outputs':full_out,
           'samples':[{'original':s['original'],'output':s['output'],'edits':s['edits']} for s in samples],
           'settings':{'C':C,'threshold':THRESHOLD,'selection':'BenchLS dev, grouped by target word; see refine_round*.json'}},
          open(V3/'evaluation.json','w'),indent=2,default=float)
