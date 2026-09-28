import math,re,pickle
from collections import Counter
from functools import lru_cache
import numpy as np
from nltk.corpus import wordnet as wn,brown,stopwords
from lemminflect import getInflection
from .features import nlp,base_features,word_features
from .data import ROOT

POS={'NOUN':'n','VERB':'v','ADJ':'a','ADV':'r'}
PROTECTED=['neural network','random forest','support vector machine','machine learning','operating system','interest rate','net income','cash flow','separation of powers','due process','confidence interval','statistical significance','gross domestic product','natural language processing','capital expenditure']
NEG={'not','no','never','neither','nor','without','cannot',"n't"}
MODAL={'may','might','must','could','should','shall','will','would','can'}

@lru_cache(1)
def stop():return set(stopwords.words('english'))

@lru_cache(40000)
def synsets(lemma,pos):return wn.synsets(lemma,pos=pos)

@lru_cache(50000)
def signature(name):
    s=wn.synset(name)
    return set(re.findall('[a-z]+',' '.join([s.definition(),*s.examples(),*s.lemma_names()]).lower()))-stop()

def generate(doc,token):
    if token is None or token.pos_ not in POS:return []
    found={}
    for s in synsets(token.lemma_.lower(),POS[token.pos_]):
        for lemma in s.lemma_names():
            if '_' in lemma or not lemma.isalpha():continue
            form=getInflection(lemma,tag=token.tag_)
            if not form:continue
            word=form[0]
            if token.text.istitle():word=word.title()
            elif token.text.isupper():word=word.upper()
            if word.lower()==token.text.lower():continue
            entry=found.setdefault(word.lower(),{'word':word,'lemma':lemma,'senses':[]})
            entry['senses'].append(s.name())
    return list(found.values())

@lru_cache(1)
def language_model():
    path=ROOT/'data/cache/brown-bigrams.pkl'
    if path.exists():
        with open(path,'rb') as f:return pickle.load(f)
    uni=Counter();bi=Counter()
    for sent in brown.sents():
        tokens=['<s>']+[w.lower() for w in sent]+['</s>']
        uni.update(tokens);bi.update(zip(tokens,tokens[1:]))
    data=(uni,bi,len(uni))
    with open(path,'wb') as f:pickle.dump(data,f)
    return data

def log_bigram(a,b):
    uni,bi,v=language_model()
    return math.log((bi[a,b]+.1)/(uni[a]+.1*v))

def local_lm(doc,token,candidate):
    left=doc[token.i-1].text.lower() if token.i else '<s>'
    right=doc[token.i+1].text.lower() if token.i+1<len(doc) else '</s>'
    return (log_bigram(left,candidate.lower())+log_bigram(candidate.lower(),right))/2

def fit_features(doc,token,candidate):
    context={t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and t.i!=token.i}
    context-=set([token.lemma_.lower()])
    senses=synsets(token.lemma_.lower(),POS.get(token.pos_,'n'))
    scored=[(len(context&signature(s.name())),s.name()) for s in senses]
    best=max((x[0] for x in scored),default=0)
    selected={name for score,name in scored if best>0 and score==best}
    shared=bool(selected&set(candidate.get('senses',[])))
    overlap=max((len(context&signature(s)) for s in candidate.get('senses',[])),default=0)
    delta=local_lm(doc,token,candidate['word'])-local_lm(doc,token,token.text)
    return {'overlap':overlap,'best_overlap':best,'sense_match':shared,'sense_tied':len(selected)>1,
            'lm_delta':float(delta),'fit_score':float(2*int(shared)+min(overlap,3)/3+np.clip(delta,-4,4)/4)}

def simple_tokens(text):return re.findall(r"[a-z]+|n't",text.lower())

def guardrails(original,candidate,original_doc=None):
    od=original_doc or nlp()(original);nd=nlp()(candidate)
    old_words=Counter(t.lower_ for t in od);new_words=Counter(t.lower_ for t in nd)
    checks={}
    checks['numbers']=Counter(re.findall(r'\d+(?:[,.]\d+)*',original))==Counter(re.findall(r'\d+(?:[,.]\d+)*',candidate))
    quantity_pattern=r'\b(\d+(?:[,.]\d+)*)\s*(kilograms?|kg|grams?|g|dollars?|percent|%|seconds?|minutes?|hours?|meters?|metres?|km|miles?)\b'
    checks['quantities']=Counter(re.findall(quantity_pattern,original.lower()))==Counter(re.findall(quantity_pattern,candidate.lower()))
    checks['negation']=Counter({k:old_words[k] for k in NEG if old_words[k]})==Counter({k:new_words[k] for k in NEG if new_words[k]})
    checks['modality']=Counter({k:old_words[k] for k in MODAL if old_words[k]})==Counter({k:new_words[k] for k in MODAL if new_words[k]})
    checks['entities']=all(e.text.lower() in candidate.lower() for e in od.ents)
    named=lambda d:{e.text.lower():(e.root.dep_,e.root.head.i) for e in d.ents if e.label_ in {'PERSON','ORG','GPE'}}
    old_named,new_named=named(od),named(nd)
    checks['entity_roles']=all(new_named.get(k)==v for k,v in old_named.items())
    checks['technical_phrases']=all(p not in original.lower() or p in candidate.lower() for p in PROTECTED)
    # For one-token substitutions, argument roles and negation attachment remain comparable by token index.
    roles={'nsubj','nsubjpass','dobj','obj','iobj','pobj'}
    if len(od)==len(nd):
        checks['argument_roles']=all(nd[t.i].dep_==t.dep_ and nd[t.i].head.i==t.head.i for t in od if t.dep_ in roles)
        checks['negation_attachment']=all(nd[t.i].head.i==t.head.i for t in od if t.lower_ in NEG)
    else:
        checks['argument_roles']=False
        checks['negation_attachment']=False
    failed=[k for k,v in checks.items() if not v]
    return {'pass':not failed,'checks':checks,'failed':failed,'semantic_certified':False}

class LexicalPipeline:
    def __init__(self,word_model,settings=None,checker=None):
        self.word_model=word_model;self.scores={};self.settings=settings or {'min_gain':.02,'min_fit':.5};self.checker=checker
    def prime(self,words):
        missing=sorted({w.lower() for w in words}-self.scores.keys())
        if missing:
            for w,p in zip(missing,self.word_model.predict([word_features(w) for w in missing])):self.scores[w]=float(p)
    def score(self,word):
        self.prime([word]);return self.scores[word.lower()]
    def rank(self,doc,token,candidates,mode='context',check=True):
        if token is None:return {'replacement':None,'candidates':[],'reason':'Target could not be aligned','changed':False}
        self.prime([token.text]+[c['word'] for c in candidates])
        original_score=self.score(token.text)
        ranked=[]
        for c in candidates:
            row={**c,'difficulty':self.score(c['word'])}
            row['gain']=original_score-row['difficulty']
            row.update(fit_features(doc,token,c))
            if self.checker is not None:
                f=context_vector(doc,token,c,row)
                row['fit_score']=float(self.checker.predict_proba([f])[0,1])
            if mode=='dictionary':
                row['eligible']=base_features(c['word'])['zipf']>base_features(token.text)['zipf']
                row['rank_score']=base_features(c['word'])['zipf']
            else:
                row['eligible']=row['gain']>=self.settings['min_gain'] and (mode=='learned' or row['fit_score']>=self.settings['min_fit'])
                row['rank_score']=row['gain']+.03*row['fit_score'] if mode in {'context','guarded'} else row['gain']
            ranked.append(row)
        ranked.sort(key=lambda x:(-x['rank_score'],x['word'].lower()))
        selected=None;rejected=[]
        for c in ranked:
            if not c['eligible']:continue
            new=doc.text[:token.idx]+c['word']+doc.text[token.idx+len(token):]
            if mode=='guarded' and check:
                g=guardrails(doc.text,new,doc);c['guardrails']=g
                if not g['pass']:rejected.extend(g['failed']);continue
            selected=c;break
        new=doc.text if selected is None else doc.text[:token.idx]+selected['word']+doc.text[token.idx+len(token):]
        return {'original':doc.text,'target':token.text,'offset':token.idx,'original_difficulty':original_score,
                'replacement':selected['word'] if selected else token.text,'output':new,'changed':selected is not None,
                'reason':'Selected eligible candidate' if selected else ('Rejected: '+', '.join(sorted(set(rejected))) if rejected else 'No candidate met the simplicity/context thresholds'),
                'candidates':ranked[:10]}

def context_vector(doc,token,candidate,features=None):
    f=features or fit_features(doc,token,candidate)
    return {'sense_match':int(f['sense_match']),'overlap':f['overlap'],'best_overlap':f['best_overlap'],
            'lm_delta':f['lm_delta'],'wordnet_member':int(bool(candidate.get('senses'))),
            'candidate_zipf':base_features(candidate['word'])['zipf'],
            'source_zipf':base_features(token.text)['zipf'],'length_difference':len(candidate['word'])-len(token.text),
            'multiword':int(' ' in candidate['word']),'pos':token.pos_}
