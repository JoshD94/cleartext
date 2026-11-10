"""Small explicit phrase inventory and conservative syntax rules."""
import re
from lemminflect import getInflection
from .features import nlp
from .lexical import NEG,MODAL
from .protected_literals import literal_spans
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
    code=literal_spans(text)
    for source,target in PHRASES:
        for m in re.finditer(r'(?<!\w)'+re.escape(source)+r'(?!\w)',text,re.I):
            if any(m.start()<span.end and m.end()>span.start for span in code):continue
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

# Negation rule (team spec, Sentence Structure Rule tab): double negatives
# and "not un-" constructions. "It is not uncommon" -> "It is common";
# "Not an uncommon behavior" -> "A common behavior"; "Not hard" -> "easy".
# The map below is curated, not mined: each pair is a genuine negative
# prefix (un/in/im/ir/il, the "in" assimilations) plus its positive stem,
# both verified as real words. Plain adjectives are a separate small
# whitelist (the doc's "hard" only); general antonym substitution is
# deliberately NOT done, because "not good" is not "bad" and "not big"
# is not "small". Note the litotes caveat: flattening strengthens the
# claim ("not uncommon" is weaker than "common"). The rule applies it
# anyway, per the team spec, and the evals preservation suite will flag
# these edits on its negation check by design; that check compares
# negation markers, not truth conditions.
NEGATION_PREFIX={
 'uncommon':'common','unusual':'usual','unhappy':'happy','unnecessary':'necessary',
 'unfamiliar':'familiar','unimportant':'important','unlikely':'likely','unable':'able',
 'unaware':'aware','unavailable':'available','uncertain':'certain','uneasy':'easy',
 'untrue':'true','unwilling':'willing','unkind':'kind','unfair':'fair','unpopular':'popular',
 'unlucky':'lucky','unsafe':'safe','unhealthy':'healthy','unhelpful':'helpful',
 'unnatural':'natural','unpleasant':'pleasant','unprepared':'prepared',
 'unreasonable':'reasonable','unremarkable':'remarkable','unsatisfactory':'satisfactory',
 'unsuccessful':'successful','unsure':'sure','unwell':'well','unwise':'wise',
 'unwelcome':'welcome','unworthy':'worthy','unclear':'clear','unfinished':'finished',
 'unfriendly':'friendly','unjust':'just','unknown':'known','unlimited':'limited',
 'unrelated':'related','unreliable':'reliable','unresolved':'resolved','unstable':'stable',
 'untested':'tested','untrained':'trained','unwanted':'wanted','unused':'used',
 'incorrect':'correct','inaccurate':'accurate','incomplete':'complete',
 'inadequate':'adequate','inappropriate':'appropriate','incapable':'capable',
 'inconvenient':'convenient','indecisive':'decisive','indirect':'direct',
 'ineffective':'effective','inefficient':'efficient','inexperienced':'experienced',
 'inflexible':'flexible','informal':'formal','infrequent':'frequent','insecure':'secure',
 'insignificant':'significant','invisible':'visible','involuntary':'voluntary',
 'invalid':'valid',
 'impossible':'possible','immoral':'moral','impatient':'patient','imperfect':'perfect',
 'impersonal':'personal','impractical':'practical','imprecise':'precise',
 'improper':'proper','immature':'mature',
 'irregular':'regular','irrelevant':'relevant','irresponsible':'responsible',
 'irrational':'rational',
 'illegal':'legal','illegible':'legible','illogical':'logical','illiterate':'literate',
}
# Plain adjectives with no negative prefix: only the doc's example. Extend deliberately, never by antonym lookup.
NEGATION_PLAIN={'hard':'easy'}
# Double negatives with a positive counterpart: "not nothing" -> "something".
# "not never" is deliberately excluded: it is ambiguous between "always" and "sometimes".
NEGATION_PRONOUN={'nothing':'something','nobody':'somebody'}
_AN_ARTICLE={'hour','honest','honor','honour','heir','herb'}
_A_ARTICLE={'university','unique','useful','user','uniform','union','unit','united','usual','utility','utensil','one','once','european','eulogy'}
def _article_for(word):
    w=word.lower()
    if w in _AN_ARTICLE:return 'an'
    if w in _A_ARTICLE:return 'a'
    return 'an' if w[:1] in 'aeiou' else 'a'
def _negate_sentence(doc,sent,code,ents):
    toks=list(sent)
    if any(t.is_quote for t in toks):return None
    if any(t.lower_ in {'neither','nor'} for t in toks):return None
    def overlaps(a,b):
        return any(a<span.end and b>span.start for span in code+ents)
    def build(neg,target,replacement,rule):
        if any(t.lower_ in {'only','just','merely','simply'} and neg.i<t.i<=target.i for t in toks):return None
        if any(t.lower_ in {'all','every','each','both','any'} and neg.i<t.i<=target.i for t in toks):return None
        if any(t.lower_=='but' and neg.i<t.i for t in toks):return None
        mid=[t for t in toks if neg.i<t.i<target.i]
        if len(mid)==1 and mid[0].lower_ in {'a','an'}:
            article_tok=mid[0]
            fixed=_article_for(replacement)
        elif not mid:
            article_tok=None
            fixed=None
        else:
            return None
        if overlaps(neg.idx,target.idx+len(target.text)):return None
        # Rebuild from tokens: dropping "not" must not swallow the space
        # after it, and dropping "n't" must not glue "is" to the next word.
        parts=[]
        pending_space=False
        first=True
        for t in toks:
            if t.i==neg.i:
                pending_space=bool(t.whitespace_)
                continue
            if article_tok is not None and t.i==article_tok.i:
                text=fixed
            elif t.i==target.i:
                text=replacement
            else:
                text=t.text
            if pending_space and parts and not parts[-1].endswith(' '):
                parts.append(' ')
            if first and neg.i==sent.start and neg.text[0].isupper():
                text=text[:1].upper()+text[1:]
            parts.append(text+t.whitespace_)
            pending_space=False
            first=False
        # The last token's trailing space belongs to the text after the
        # sentence, not the sentence itself; drop it so reassembly does
        # not double it.
        new=''.join(parts).rstrip()
        return new,{'stage':'negation','rule':rule,'before':sent.text,'after':new}
    for neg in [t for t in toks if t.dep_=='neg' and t.lower_ in {'not',"n't"}]:
        head=neg.head
        # A0: the negated word itself is the adjective ("Not uncommon.").
        if head.pos_=='ADJ' and head.tag_=='JJ':
            rep=NEGATION_PREFIX.get(head.lower_) or NEGATION_PLAIN.get(head.lower_)
            if rep and not any(c.dep_=='advmod' for c in head.children):
                r=build(neg,head,rep,'not-un-prefix' if head.lower_ in NEGATION_PREFIX else 'not-plain')
                if r:return r
        # A1: copula predicate adjective ("It is not uncommon.").
        # Lemma "be" only: the parser tags existential "there is" as VERB.
        if head.lemma_=='be':
            for c in head.children:
                if c.dep_ in {'acomp','xcomp','oprd'} and c.pos_=='ADJ' and c.tag_=='JJ':
                    rep=NEGATION_PREFIX.get(c.lower_) or NEGATION_PLAIN.get(c.lower_)
                    if rep and not any(x.dep_=='advmod' for x in c.children):
                        r=build(neg,c,rep,'not-un-prefix' if c.lower_ in NEGATION_PREFIX else 'not-plain')
                        if r:return r
                # C: negated pronoun complement ("It is not nothing.").
                if c.dep_ in {'attr','nsubj'} and c.pos_=='PRON' and c.lower_ in NEGATION_PRONOUN:
                    r=build(neg,c,NEGATION_PRONOUN[c.lower_],'double-negative')
                    if r:return r
                # A2/D: noun complement ("That is not an uncommon behavior." / "There is not no reason.").
                if c.dep_ in {'attr','nsubj','nsubjpass'} and c.pos_ in {'NOUN','PROPN'}:
                    for a in c.children:
                        if a.dep_=='amod' and a.pos_=='ADJ' and a.tag_=='JJ':
                            rep=NEGATION_PREFIX.get(a.lower_) or NEGATION_PLAIN.get(a.lower_)
                            if rep and not any(x.dep_=='advmod' for x in a.children):
                                r=build(neg,a,rep,'not-un-prefix' if a.lower_ in NEGATION_PREFIX else 'not-plain')
                                if r:return r
                        if a.dep_=='det' and a.lower_=='no':
                            r=build(neg,a,'some','double-negative')
                            if r:return r
        # A2 fragment: the negated noun is the head itself ("Not an uncommon behavior.").
        if head.pos_ in {'NOUN','PROPN'}:
            for a in head.children:
                if a.dep_=='amod' and a.pos_=='ADJ' and a.tag_=='JJ':
                    rep=NEGATION_PREFIX.get(a.lower_) or NEGATION_PLAIN.get(a.lower_)
                    if rep and not any(x.dep_=='advmod' for x in a.children):
                        r=build(neg,a,rep,'not-un-prefix' if a.lower_ in NEGATION_PREFIX else 'not-plain')
                        if r:return r
    return None
def negation_rewrite(text):
    """Flatten negation, at most one edit per sentence. The negated element
    must be the predicate (copula complement, fragment noun phrase, or the
    adjective itself); negation over a verb with the adjective buried in an
    object ("did not give an uncommon answer") is left alone, as are
    correlatives ("not only"), degree-modified adjectives ("not very
    uncommon"), quoted text, code literals, and named entities."""
    doc=nlp()(text)
    code=literal_spans(text)
    ents=list(doc.ents)
    out=text
    edits=[]
    for sent in reversed(list(doc.sents)):
        r=_negate_sentence(doc,sent,code,ents)
        if not r:continue
        new,edit=r
        out=out[:sent.start_char]+new+out[sent.end_char:]
        edits.append(edit)
    edits.reverse()
    return out,edits
