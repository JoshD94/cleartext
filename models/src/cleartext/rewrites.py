"""Small explicit phrase inventory and conservative syntax rules."""
import re
from lemminflect import getInflection
from .features import nlp
from .lexical import NEG,MODAL
# Authored prototype inventory, not mined from PPDB or a trained phrase generator.
PHRASES=[
 ('in the event that','if'),('despite the fact that','although'),
 ('at the present time','now'),('with the exception of','except'),
 ('a large number of','many'),('on a daily basis','daily'),
 ('in close proximity to','near'),('in order to','to'),
 ('prior to','before'),('subsequent to','after'),
]
def lower_article(s):
    return s[:1].lower()+s[1:] if s.startswith(('The ','A ','An ')) else s
def initial(s):
    return s[:1].upper()+s[1:]
def structural(text):
    doc=nlp()(text);edits=[]
    # Only independent, fully expressed clauses joined by comma + and.
    if len(list(doc.sents))!=1 or not text.rstrip().endswith(".") or any(t.is_quote for t in doc):return text,edits
    forbidden={'if','unless','although','because','while','when','until','not','never','no','without','may','might','must','could','should','can'}
    forbidden |= NEG | MODAL
    if any(t.lower_ in forbidden or t.dep_ in {'neg','relcl','advcl','ccomp'} for t in doc):return text,edits
    roots=[t for t in doc if t.dep_=='ROOT']
    if not roots:return text,edits
    root=roots[0]
    for t in doc:
        if t.lower_=='and' and t.i>0 and doc[t.i-1].text==',':
            right=doc[t.i+1:]
            predicates=[x for x in right if x.dep_=='conj' and x.head==root and any(c.dep_ in {'nsubj','nsubjpass'} for c in x.children)]
            if len(predicates)!=1:continue
            left=text[:doc[t.i-1].idx].strip();righttext=text[t.idx+len(t):].strip()
            if not any(c.dep_ in {'nsubj','nsubjpass'} for c in root.children):continue
            output=left+'. '+initial(righttext)
            return output,[{'stage':'structure','rule':'independent clauses','before':text,'after':output}]
    # Passive only: one simple be auxiliary, explicit by-agent, no extra auxiliaries or clauses.
    aux=[t for t in root.children if t.dep_=='auxpass']
    subjects=[t for t in root.children if t.dep_=='nsubjpass']
    agents=[t for t in root.children if t.dep_=='agent' and t.lower_=='by']
    if len(aux)!=1 or len(subjects)!=1 or len(agents)!=1:return text,edits
    if aux[0].lower_ not in {'was','were','is','are'} or root.tag_!='VBN':return text,edits
    if any(t.dep_ in {'aux','conj','xcomp','ccomp','relcl'} for t in doc):return text,edits
    objects=[t for t in agents[0].children if t.dep_=='pobj']
    if len(objects)!=1:return text,edits
    actor=objects[0];patient=subjects[0]
    actor_tokens=list(actor.subtree);patient_tokens=list(patient.subtree)
    covered={root.i,aux[0].i,agents[0].i}|{t.i for t in actor_tokens+patient_tokens}
    if any(not t.is_punct and t.i not in covered for t in doc):return text,edits
    if actor.pos_=='PRON' or patient.pos_=='PRON':return text,edits
    tag='VBD' if aux[0].lower_ in {'was','were'} else ('VBP' if actor.tag_=='NNS' else 'VBZ')
    forms=getInflection(root.lemma_,tag=tag)
    if not forms:return text,edits
    span=lambda ts:doc[min(t.i for t in ts):max(t.i for t in ts)+1].text
    output=initial(span(actor_tokens))+' '+forms[0]+' '+lower_article(span(patient_tokens))+'.'
    return output,[{'stage':'structure','rule':'explicit-agent passive','before':text,'after':output}]

def phrase_rewrite(text,complexity):
    matches=[]
    doc=nlp()(text)
    for source,target in PHRASES:
        for m in re.finditer(r'(?<!\w)'+re.escape(source)+r'(?!\w)',text,re.I):
            if any(m.start()<e.end_char and m.end()>e.start_char for e in doc.ents):continue
            old=complexity(source);new=complexity(target)
            matches.append({'start':m.start(),'end':m.end(),'source':m.group(),'replacement':initial(target) if m.group()[0].isupper() else target,
                            'before_complexity':old,'after_complexity':new,'gain':old-new,'stage':'phrase',
                            'complexity_note':'CompLex word/phrase regression; long expressions extrapolate beyond mostly two-word training targets.'})
    # One phrase edit; favor greatest predicted reduction. Meanings supplied by authored rules.
    matches=[m for m in matches if m['gain']>=.02]
    if not matches:return text,[]
    best=max(matches,key=lambda m:m['gain'])
    return text[:best['start']]+best['replacement']+text[best['end']:],[best]
