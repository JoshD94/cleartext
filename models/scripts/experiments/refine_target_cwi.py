"""Which word gets the edit? Measured with CWI 2018 per-word annotations (share of 20 annotators marking the word hard).

For each sentence, every content word the detector could flag is ranked by the v5 pipeline once and cached. Choice
rules are then compared offline: share of edits on words most annotators found hard (>= 0.5), mean hardness of edited
words, and edit count. Chosen on CWI dev; CWI test reported once.
"""
import sys,os,csv,json,pickle,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from cleartext.data import ROOT,RAW
from cleartext.features import parse
from cleartext.ensemble_pipeline import EnsembleClearText,CONTENT,LATEST
pipe=EnsembleClearText.load(LATEST)

def detect_prob(doc,tok):return float(pipe.hardness(doc,[tok])[0])

def cache(split,limit=None):
    path=ROOT/f"data/cache/refine-cwi-targets-{os.environ.get('CWI_CACHE','v1')}-{split}.pkl"
    if path.exists():return pickle.load(open(path,'rb'))
    raw=list(csv.reader(open(RAW/f'cwi_english_{split}.tsv'),delimiter='\t',quoting=csv.QUOTE_NONE))
    hard=collections.defaultdict(dict)
    for r in raw:
        if ' ' not in r[4].strip():hard[r[1]][int(r[2])]=float(r[10])
    sents=sorted(hard)[:limit];docs=parse(sents,f'cwi-{split}-full');out=[]
    for k,(s,doc) in enumerate(zip(sents,docs)):
        words=[]
        for tok in doc:
            if not tok.is_alpha or tok.ent_type_ or tok.pos_ not in CONTENT or tok.is_stop:continue
            p=detect_prob(doc,tok)
            if p<.2:continue
            sel=pipe.rank_word(doc,tok)['selected']
            words.append({'word':tok.text,'pos':tok.pos_,'detect':p,'hardness':hard[s].get(tok.idx),
                          'replacement':sel['word'] if sel else None,'accept':sel['accept'] if sel else None})
        out.append({'text':s,'words':words})
        if k%100==0:print('CWI',split,k,'of',len(sents),flush=True)
    pickle.dump(out,open(path,'wb'));return out

def run(cases,key,detect_t,noun_t):
    """noun_t: nouns need detector probability >= noun_t (plain nouns like 'database' are flagged weakly)."""
    edited=[]
    for case in cases:
        ok=[w for w in case['words'] if w['replacement'] and w['detect']>=detect_t and (w['pos']!='NOUN' or w['detect']>=noun_t)]
        if ok:edited.append(max(ok,key=key))
    known=[w for w in edited if w['hardness'] is not None]
    return {'sentences':len(cases),'edits':len(edited),'annotated':len(known),
            'on_hard_word':round(float(np.mean([w['hardness']>=.5 for w in known])),3) if known else 0.,
            'mean_hardness':round(float(np.mean([w['hardness'] for w in known])),3) if known else 0.,
            'hard_edits':int(sum(w['hardness']>=.5 for w in known))}

S={s:cache(s) for s in ['Dev','Test']}
KEYS={'decision':lambda w:w['accept'],'detector':lambda w:w['detect'],'product':lambda w:w['accept']*w['detect']}
configs=[(k,d,n) for k in KEYS for d in [.39,.5,.6] for n in [0.,.5,.6,.7,.8]]
dev={c:run(S['Dev'],KEYS[c[0]],c[1],c[2]) for c in configs}
current=('decision',.39,0.)
# Objective: number of edits on words most annotators marked hard, minus edits on words most did not.
score=lambda r:2*r['hard_edits']-r['annotated']
best=max(configs,key=lambda c:score(dev[c]))
for c in [current,best]+[x for x in configs if x[2]==0. and x[1]==.39]:
    t=run(S['Test'],KEYS[c[0]],c[1],c[2])
    print(f"{str(c):<28} dev: {dev[c]['edits']:>3} edits, {dev[c]['on_hard_word']:.2f} on hard words, net {score(dev[c]):>4}"
          f" | test: {t['edits']:>3} edits, {t['on_hard_word']:.2f} on hard words, net {score(t):>4}{'  <- chosen on dev' if c==best else ''}")
json.dump({'dev':{str(k):v for k,v in dev.items()},'chosen':list(best),'test_chosen':run(S['Test'],KEYS[best[0]],best[1],best[2]),
           'test_current':run(S['Test'],KEYS[current[0]],current[1],current[2])},open(LATEST/'refine_target_cwi.json','w'),indent=2)
