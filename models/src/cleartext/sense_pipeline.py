"""SemCor sense filtering with original-context tracing."""
import pickle
from .data import ROOT
from .pipeline import ClearText
from .context_pipeline import ContextClearText
from .wsd import SenseModel
from .lexical import generate,guardrails
class SenseClearText(ContextClearText):
    def __init__(self):
        self.base=ClearText()
        self.phrase_model=pickle.load(open(ROOT/'runs/context-20260924/phrase_model.pkl','rb'))
        self.senses=SenseModel.load()
    def rank_word(self,doc,token,original_doc=None,guard=True,sense_filter=True):
        original_doc=original_doc if original_doc is not None else doc
        originals=[t for t in original_doc if t.text==token.text and t.lemma_==token.lemma_]
        context_token=originals[0] if len(originals)==1 else token
        context_doc=original_doc if len(originals)==1 else doc
        prediction=self.senses.from_token(context_doc,context_token)
        candidates=generate(doc,token);scorer=self.base.lexical
        scorer.prime([token.text]+[c['word'] for c in candidates])
        rows=[]
        for c in candidates:
            gain=scorer.score(token.text)-scorer.score(c['word']);reasons=[]
            if gain<.02:reasons.append('not sufficiently simpler')
            if sense_filter:
                if not prediction['trained'] and len(prediction['ranked'])>1:reasons.append('unseen ambiguous lemma')
                if prediction['sense'] not in c['senses']:reasons.append('different selected meaning')
            output=doc.text[:token.idx]+c['word']+doc.text[token.idx+len(token):]
            if not reasons and guard:
                g=guardrails(doc.text,output,doc)
                if not g['pass']:reasons+=g['failed']
            rows.append({**c,'gain':gain,'fit':None,'rejections':reasons,'output':output})
        accepted=sorted([r for r in rows if not r['rejections']],key=lambda r:(-r['gain'],r['word'].lower()))
        return {'target':token.text,'start':token.idx,'difficulty':scorer.score(token.text),
                'sense_prediction':prediction,'candidates':rows,'selected':accepted[0] if accepted else None}
