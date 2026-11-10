"""End-to-end round: targets are NOT given. Which flagged word should receive the single edit?

A sentence counts as correct when the pipeline edits the dataset's target word with a gold substitute.
Edits to other words are counted separately (not verifiable against these labels).
"""
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cleartext.data import ROOT,benchls_data,tsar_data
from cleartext.features import parse,target_token
from cleartext.ensemble_pipeline import EnsembleClearText,CONTENT
import os
V3=ROOT/os.environ.get('TARGET_RUN','runs/ensemble-v3-20260927');CACHE=os.environ.get('TARGET_CACHE','v1')
pipe=EnsembleClearText.load(V3)
def detect_prob(doc,tok):return float(pipe.hardness(doc,[tok])[0])

def cache(name,rows,docs,targets):
    path=ROOT/f'data/cache/refine-targets-{CACHE}-{name}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    out=[]
    for k,(r,doc,tgt) in enumerate(zip(rows,docs,targets)):
        words=[]
        for tok in doc:
            if not tok.is_alpha or tok.ent_type_ or tok.pos_ not in CONTENT or tok.is_stop:continue
            p=detect_prob(doc,tok)
            if p<.2:continue
            rank=pipe.rank_word(doc,tok);sel=rank['selected']
            words.append({'i':tok.i,'word':tok.text,'pos':tok.pos_,'detect':p,'is_target':tgt is not None and tok.i==tgt.i,
                          'replacement':sel['word'] if sel else None,'accept':sel['accept'] if sel else None,
                          'gold':bool(sel and sel['word'].lower() in r['gold'])})
        out.append({'id':r['id'],'words':words,'target_found':tgt is not None})
        if k%100==0:print('TARGETS',name,k,'of',len(rows),flush=True)
    pickle.dump(out,open(path,'wb'));return out

b=benchls_data();S={}
for split in ['dev','holdout']:
    rows=b[split];docs=parse([r['text'] for r in rows],f'benchls-{split}')
    tg=[target_token(d,r['target'],sum(len(w)+1 for w in r['text'].split(' ')[:r['index']])) for r,d in zip(rows,docs)]
    S[f'benchls_{split}']=cache(f'benchls-{split}',rows,docs,tg)
rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test')
S['tsar_test']=cache('tsar-test',rows,docs,[target_token(d,r['target']) for r,d in zip(rows,docs)])

def run(cases,key,detect_t):
    edited=on_target=correct=0
    for case in cases:
        ok=[w for w in case['words'] if w['detect']>=detect_t and w['replacement'] is not None]
        if not ok:continue
        w=max(ok,key=key);edited+=1;on_target+=w['is_target'];correct+=w['is_target'] and w['gold']
    return {'sentences':len(cases),'edited':edited,'on_target':on_target,'correct':correct,
            'correct_rate':round(correct/len(cases),3),'on_target_rate':round(on_target/max(edited,1),3)}
KEYS={'decision probability':lambda w:w['accept'],'detector probability':lambda w:w['detect'],'product':lambda w:w['accept']*w['detect']}
thresholds=[.39,.5,.6,.7]
out={}
for kname,key in KEYS.items():
    for t in thresholds:
        out[kname,t]={s:run(c,key,t) for s,c in S.items()}
best=max(out,key=lambda k:out[k]['benchls_dev']['correct'])
for k,v in out.items():
    mark='  <- chosen on dev' if k==best else ''
    print(f"{k[0]:<22} detect>={k[1]:<4}  dev correct {v['benchls_dev']['correct']:>3} (on target {v['benchls_dev']['on_target_rate']:.2f})"
          f" | holdout {v['benchls_holdout']['correct']:>3} | tsar {v['tsar_test']['correct']:>3}{mark}")
json.dump({'results':{f'{k[0]}|{k[1]}':v for k,v in out.items()},'chosen':list(best)},open(V3/'refine_target_choice.json','w'),indent=2)
