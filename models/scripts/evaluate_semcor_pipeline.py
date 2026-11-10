import sys,json,gzip,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nltk.corpus import wordnet as wn
from sklearn.metrics import precision_score,recall_score,f1_score
from cleartext.data import ROOT,RAW,tsar_data
from cleartext.features import parse,target_token
from cleartext.lexical import POS,synsets
from cleartext.sense_pipeline import SenseClearText
run=ROOT/'runs/semcor-20260924';started=time.time();pipeline=SenseClearText()
rows=tsar_data('test');docs=parse([r['text'] for r in rows],'tsar-test')
results=[];outputs={}
for name,guard,sense_filter in [('basic',False,False),('guards_only',True,False),('sense_only',False,True),('sense_and_guards',True,True)]:
    out=[];lines=[]
    for row,doc in zip(rows,docs):
        token=target_token(doc,row['target'])
        rank=pipeline.rank_word(doc,token,guard=guard,sense_filter=sense_filter)
        answer=rank['selected']['word'] if rank['selected'] else row['target']
        text=rank['selected']['output'] if rank['selected'] else row['text']
        changed=text!=row['text']
        out.append({'id':row['id'],'original':row['text'],'target':row['target'],'replacement':answer,'output':text,
                    'changed':changed,'gold_match':bool(changed and answer.lower() in row['gold']),
                    'sense':rank['sense_prediction'],'candidates':rank['candidates']})
        lines.append(row['text']+'\t'+row['target']+'\t'+answer.lower())
    path=run/f'tsar_{name}.tsv';path.write_text('\n'.join(lines)+'\n')
    official=run/f'tsar_{name}_official.txt'
    subprocess.run([sys.executable,str(RAW/'tsar_eval.py'),'--gold_file',str(RAW/'tsar_test.tsv'),'--predictions_file',str(path),'--output_file',str(official)],check=True,capture_output=True)
    edits=sum(r['changed'] for r in out);correct=sum(r['gold_match'] for r in out)
    metric={'name':name,'n':len(out),'edits':edits,'correct':correct,'top1':correct/len(out),
            'edit_rate':edits/len(out),'match_among_edits':correct/edits if edits else 0.,'official':official.read_text()}
    outputs[name]=out;results.append(metric);print('TSAR',json.dumps({k:v for k,v in metric.items() if k!='official'}),flush=True)
# Existing annotations provide a second, separate check of contextual suitability.
d=json.load(gzip.open(RAW/'swords_test.json.gz','rt'))
ids=list(d['contexts']);docs=parse([d['contexts'][i]['context'] for i in ids],'swords-test');byid=dict(zip(ids,docs))
targets={}
for tid,t in d['targets'].items():
    doc=byid[t['context_id']];token=target_token(doc,t['target'],t['offset'])
    if token is None:continue
    prediction=pipeline.senses.from_token(doc,token)
    source_senses=synsets(token.lemma_.lower(),POS.get(token.pos_,'n'))
    targets[tid]=(prediction,source_senses,POS.get(token.pos_,'n'))
labels=[];base=[];filtered=[]
for sid,c in d['substitutes'].items():
    if c['target_id'] not in targets:continue
    votes=d['substitute_labels'][sid]
    if not votes:continue
    pred,senses,pos=targets[c['target_id']]
    candidate=c['substitute'].lower().replace(' ','_')
    forms={candidate,wn.morphy(candidate,pos)}
    matches={s.name() for s in senses if forms&{l.lower() for l in s.lemma_names()}}
    labels.append(votes.count('TRUE')/len(votes)>=.5)
    base.append(bool(matches))
    filtered.append(bool(pred['sense'] in matches and (pred['trained'] or len(pred['ranked'])==1)))
swords=[]
for name,pred in [('any_wordnet_meaning',base),('selected_semcor_meaning',filtered)]:
    metric={'name':name,'n':len(labels),'accepted':sum(pred),'precision':float(precision_score(labels,pred,zero_division=0)),
            'recall':float(recall_score(labels,pred,zero_division=0)),'f1':float(f1_score(labels,pred,zero_division=0))}
    swords.append(metric);print('SWORDS',json.dumps(metric),flush=True)
old=json.loads((ROOT/'runs/initial-20260924/basic_examples_20.json').read_text())
samples=[]
for r in old:
    samples.append({'baseline':r['output'],**pipeline.analyze(r['original'],structure=False,phrases=False)})
# Separate composed-pipeline examples keep word effects distinguishable from phrase/syntax rules.
composed=[pipeline.analyze(s) for s in [
 'Despite the fact that it rained, the match continued.',
 'The report was published by the committee.',
 'The engineer tested the software, and the manager reviewed the report.',
 'The report was published by the committee, and a large number of students read it.']]
banks=[]
for s in ['She deposited money at the bank.','They sat on the bank beside the river.']:
    doc=parse([s],'sense-bank')[0];tok=next(t for t in doc if t.text=='bank')
    banks.append({'text':s,**pipeline.senses.from_token(doc,tok)})
report={'tsar':results,'swords':swords,'outputs':outputs,'samples':samples,'composed':composed,'banks':banks,
        'note':'SWORDS and TSAR are reused test benchmarks from prior experiments, not fresh final tests. No labels from either fitted or selected the SemCor model. SWORDS figures measure dictionary candidate membership, not a full simplification pipeline.',
        'runtime_seconds':time.time()-started}
(run/'evaluation.json').write_text(json.dumps(report,indent=2))
for i,r in enumerate(samples,1):print('SAMPLE',i,r['original'],'=>',r['output'],flush=True)
for r in banks:print('BANK',r['text'],r['sense'],r['margin'],flush=True)
print('COMPLETE',flush=True)
