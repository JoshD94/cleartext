import sys,json,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cleartext.data import ROOT,download,word_data,sentence_data
from cleartext.features import parse,target_token,word_features,sentence_features
from cleartext.models import fit_compare

run=ROOT/'runs/initial-20260924';run.mkdir(parents=True,exist_ok=True)
started=time.time()
manifest=download(); words=word_data();sentences,removed=sentence_data()
print('Data loaded', {s:len(r) for s,r in sentences.items()}, 'overlap removed',removed,flush=True)
word_features_all={};context_features_all={};alignment={}
for split,rows in words.items():
    docs=parse([r['text'] for r in rows],f'complex-{split}')
    word_features_all[split]=[word_features(r['token']) for r in rows]
    context_features_all[split]=[word_features(r['token'],d,target_token(d,r['token']),True) for r,d in zip(rows,docs)]
    alignment[split]={'missing':sum(target_token(d,r['token']) is None for r,d in zip(rows,docs)),
                      'repeated_target':sum(sum(t.text.lower()==r['token'].lower() for t in d)>1 for r,d in zip(rows,docs))}
    print('Word features',split,len(rows),flush=True)
base,word_result=fit_compare(word_features_all,words,run,'word')
context,context_result=fit_compare(context_features_all,words,run,'context')
intermediate={'word':word_result,'context':context_result}
(run/'training_partial.json').write_text(json.dumps(intermediate))
sentence_features_all={}
for split,rows in sentences.items():
    docs=parse([r['text'] for r in rows],f'cefr-{split}')
    sentence_features_all[split]=sentence_features(docs,base)
    print('Sentence features',split,len(rows),flush=True)
sentence_model,sentence_result=fit_compare(sentence_features_all,sentences,run,'sentence','sentence')
counts={s:len(r) for s,r in sentences.items()}
result={'word':word_result,'context':context_result,'sentence':sentence_result,'manifest':manifest,
        'alignment':alignment,'sentence_overlap_removed':removed,
        'feature_missing':{s:{'frequency':sum(x['frequency_missing'] for x in fs),'syllables':sum(x['syllables_estimated'] for x in fs)} for s,fs in word_features_all.items()},
        'host':platform.node(),'python':platform.python_version(),'duration_seconds':time.time()-started,'seed':4701,
        'word_features':list(word_features_all['train'][0]),'context_features':list(context_features_all['train'][0]),'sentence_features':list(sentence_features_all['train'][0]),
        'packages':(ROOT/'requirements-lock.txt').read_text()}
(run/'training.json').write_text(json.dumps(result,indent=2))
print('TRAIN_COMPLETE',round(time.time()-started,1),flush=True)
