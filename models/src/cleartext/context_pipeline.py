"""Experimental v2 pipeline; keeps the original model and artifacts intact."""
import json,pickle
import numpy as np
from .data import ROOT
from .features import nlp
from .pipeline import ClearText
from .lexical import generate,guardrails,PROTECTED
from .semantics import sense_scores,vector_features
from .rewrites import structural,phrase_rewrite
from .phrase_complexity import phrase_features

class ContextClearText:
    def __init__(self):
        self.base=ClearText()
        run=ROOT/'runs/context-20260924'
        self.checker=pickle.load(open(run/'checker.pkl','rb'))
        self.config=json.loads((run/'selection.json').read_text())
        self.phrase_model=pickle.load(open(run/'phrase_model.pkl','rb'))
    def complexity(self,text):
        return float(np.clip(self.phrase_model.predict([phrase_features(text)])[0],0,1))
    def rank_word(self,doc,token,original_doc=None):
        original_doc=original_doc if original_doc is not None else doc
        originals=[t for t in original_doc if t.text==token.text and t.lemma_==token.lemma_]
        context_token=originals[0] if len(originals)==1 else token
        context_doc=original_doc if len(originals)==1 else doc
        senses=sense_scores(context_doc,context_token)
        candidates=generate(doc,token);scorer=self.base.lexical
        scorer.prime([token.text]+[c['word'] for c in candidates])
        feats=[vector_features(context_doc,context_token,c,senses=senses) for c in candidates]
        probabilities=self.checker.predict_proba(feats)[:,1] if feats else []
        rows=[]
        for c,p in zip(candidates,probabilities):
            gain=scorer.score(token.text)-scorer.score(c['word'])
            reasons=[]
            if gain<.02:reasons.append('not sufficiently simpler')
            if p<self.config['threshold']:reasons.append('weak context fit')
            # Strong gloss evidence narrows to one sense; weak evidence is left to trained checker.
            if len(senses)>1 and senses[0]['similarity']>=.25 and senses[0]['similarity']-senses[1]['similarity']>=.04 and senses[0]['sense'] not in c['senses']:
                reasons.append('different word sense')
            output=doc.text[:token.idx]+c['word']+doc.text[token.idx+len(token):]
            if not reasons:
                check=guardrails(doc.text,output,doc)
                if not check['pass']:reasons+=check['failed']
            rows.append({**c,'gain':gain,'fit':float(p),'rejections':reasons,'output':output})
        accepted=sorted([r for r in rows if not r['rejections']],key=lambda r:(-r['gain'],-r['fit'],r['word']))
        return {'target':token.text,'start':token.idx,'difficulty':scorer.score(token.text),
                'senses':senses[:3],'candidates':rows,'selected':accepted[0] if accepted else None}
    def analyze(self,text,structure=True,phrases=True,words=True):
        original_doc=nlp()(text);current=text;edits=[];trace=[]
        if structure:
            current,changes=structural(current);edits+=changes
        if phrases:
            current,changes=phrase_rewrite(current,self.complexity);edits+=changes
        if words:
            doc=nlp()(current)
            for token in doc:
                if not token.is_alpha or token.ent_type_ or token.pos_=='PROPN':continue
                if any(p in current.lower() and any(a<=token.idx<b for a,b in self._spans(current,p)) for p in PROTECTED):continue
                if self.base.lexical.score(token.text)<.3:continue
                trace.append(self.rank_word(doc,token,original_doc))
            accepted=[r for r in trace if r['selected']]
            if accepted:
                best=max(accepted,key=lambda r:r['selected']['gain'])
                current=best['selected']['output']
                edits.append({'stage':'word','source':best['target'],'replacement':best['selected']['word'],
                              'fit':best['selected']['fit'],'gain':best['selected']['gain']})
        return {'original':text,'output':current,'edits':edits,'trace':trace,
                'limits':'At most one syntax rule, one phrase and one word edit. Meanings are not certified.'}
    @staticmethod
    def _spans(text,phrase):
        import re
        return [(m.start(),m.end()) for m in re.finditer(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',text,re.I)]
