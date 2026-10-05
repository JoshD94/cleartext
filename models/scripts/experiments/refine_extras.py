"""Extra per-candidate signals from existing blocks, cached by (case id, word):

- detector: CWI detector probability for the target and for the candidate in the target's position
- sentence: change in predicted CEFR level (sentence difficulty model) after the substitution
- antonym: WordNet lists the candidate as an antonym of the target lemma
"""
import sys,pickle,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from nltk.corpus import wordnet as wn
from cleartext.data import ROOT,benchls_data,tsar_data
from cleartext.features import parse,target_token,word_features,sentence_features,nlp
from cleartext.generation import WordNetGenerator
from cleartext.ensemble_pipeline import substitute
from cleartext.wsd import word_pos
from cleartext import detection as D
load=lambda p:pickle.load(open(p,'rb'))
word_model=load(ROOT/'runs/initial-20260924/word_model.pkl');context_model=load(ROOT/'runs/initial-20260924/context_model.pkl')
sentence_model=load(ROOT/'runs/initial-20260924/sentence_model.pkl');detector=load(ROOT/'runs/ensemble-v3-20260927/detector.pkl')
generator=WordNetGenerator(('synonym','hypernym','similar'),multiword=True,fallback=True)

def detector_probs(doc,tok,words,lemmas):
    """Detector probability for each word read in the target token's position."""
    rows=D.rows_for(doc,[tok]*len(words),word_model.predict([word_features(w) for w in words]),context_model,words,lemmas)
    return D.probabilities(detector,rows)

def antonyms(lemma,pos):
    out=set()
    for s in wn.synsets(lemma,word_pos(pos)):
        for l in s.lemmas():
            if l.name().lower()==lemma:out|={a.name().lower().replace('_',' ') for a in l.antonyms()}
    return out

def extras(name,rows,docs,tokens):
    path=ROOT/f'data/cache/refine-extras-v1-{name}.pkl'
    if path.exists():return pickle.load(open(path,'rb'))
    out={};started=time.time()
    for k,(row,doc,tok) in enumerate(zip(rows,docs,tokens)):
        if tok is None:continue
        target,cands=generator.generate(tok)
        if not cands:continue
        words=[tok.text]+[c['word'] for c in cands];lemmas=[target[0] if target else tok.lemma_]+[c['lemma'] for c in cands]
        det=detector_probs(doc,tok,words,lemmas)
        new_docs=list(nlp().pipe([substitute(doc,tok,c['word']) for c in cands]))
        levels=sentence_model.predict(sentence_features([doc]+new_docs,word_model))
        ants=antonyms(target[0],target[1]) if target else set()
        for j,c in enumerate(cands):
            out[row['id'],c['word']]={'det_target':float(det[0]),'det_cand':float(det[j+1]),'sent_delta':float(levels[j+1]-levels[0]),
                                      'antonym':float(c['lemma'].lower() in ants)}
        if k%100==0:print('EXTRAS',name,k,'of',len(rows),round(time.time()-started),'s',flush=True)
    pickle.dump(out,open(path,'wb'));return out

b=benchls_data()
for split in ['dev','holdout']:
    rows=b[split];docs=parse([r['text'] for r in rows],f'benchls-{split}')
    tokens=[target_token(d,r['target'],sum(len(w)+1 for w in r['text'].split(' ')[:r['index']])) for r,d in zip(rows,docs)]
    extras(f'benchls-{split}',rows,docs,tokens)
rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test')
extras('tsar-test',rows,docs,[target_token(d,r['target']) for r,d in zip(rows,docs)])
print('COMPLETE',flush=True)
