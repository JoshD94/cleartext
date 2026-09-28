"""Inference only. Loads frozen models; never fits on review input."""
import json,pickle
from .data import ROOT
from .features import nlp,sentence_features,base_features
from .lexical import LexicalPipeline,generate

class ClearText:
    def __init__(self,run=None):
        run=run or ROOT/'runs/initial-20260924'
        def load(name):
            with open(run/name,'rb') as f:return pickle.load(f)
        self.word=load('word_model.pkl');self.sentence=load('sentence_model.pkl')
        fit=json.loads((run/'fit_selection.json').read_text())
        self.lexical=LexicalPipeline(self.word,{'min_gain':.02,'min_fit':fit['threshold']},load('fit_model.pkl'))
    def analyze(self,text,mode='guarded',difficulty_threshold=.3):
        doc=nlp()(text)
        tokens=[t for t in doc if t.is_alpha]
        self.lexical.prime([t.text for t in tokens])
        scores=[{'word':t.text,'start':t.idx,'end':t.idx+len(t),'difficulty':self.lexical.score(t.text),'pos':t.pos_} for t in tokens]
        original_level=float(self.sentence.predict(sentence_features([doc],self.word))[0])
        choices=[]
        for t in tokens:
            flagged=base_features(t.text)['zipf']<3.5 if mode=='dictionary' else self.lexical.score(t.text)>=difficulty_threshold
            if not flagged or mode=='unchanged':continue
            result=self.lexical.rank(doc,t,generate(doc,t),mode)
            if result['changed']:choices.append(result)
        # A single accepted edit per input prevents interacting replacements in this initial version.
        best=max(choices,key=lambda r:r['original_difficulty']-self.lexical.score(r['replacement'])) if choices else None
        output=best['output'] if best else text
        output_level=float(self.sentence.predict(sentence_features([nlp()(output)],self.word))[0]) if best else original_level
        return {'original':text,'output':output,'tokens':scores,'sentence_difficulty':original_level,
                'output_sentence_difficulty':output_level,'edit':best,'mode':mode,'difficulty_threshold':difficulty_threshold,
                'threshold_status':'Initial heuristic, not a validated human difficulty cutoff','max_edits':1}
