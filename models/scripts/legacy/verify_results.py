import sys,json,hashlib,gzip
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from cleartext.data import ROOT
run=ROOT/'runs/initial-20260924';t=json.loads((run/'training.json').read_text());l=json.loads((run/'lexical.json').read_text())
for d in t['manifest']:
    assert hashlib.sha256((ROOT/'data/raw'/d['file']).read_bytes()).hexdigest()==d['sha256']
for kind in ['word','context','sentence']:
    block=t[kind];winner=min(block['results'],key=lambda r:r['dev']['mae'])
    assert winner['name']==block['selected']
    rows=block['test_predictions']
    mae=np.mean([abs(r['prediction']-r['label']) for r in rows])
    assert abs(mae-winner['test']['mae'])<1e-12
    lo,hi=(1,6) if kind=='sentence' else (0,1)
    assert all(lo<=r['prediction']<=hi for r in rows)
for row in l['tsar']:
    assert row['official_evaluator_exit']==0
    pred=l['outputs'][row['name']]
    precision=sum(r['changed'] and r['replacement'].lower() in r['gold'] for r in pred)/len(pred)
    assert abs(precision-row['official_precision1'])<.00011
    assert len(pred)==373
    if row['name']=='unchanged':assert all(r['original']==r['output'] for r in pred)
a=json.load(gzip.open(ROOT/'data/raw/swords_dev.json.gz','rt'))
b=json.load(gzip.open(ROOT/'data/raw/swords_test.json.gz','rt'))
normalize=lambda s:' '.join(s.lower().split())
assert not ({normalize(c['context']) for c in a['contexts'].values()} & {normalize(c['context']) for c in b['contexts'].values()})
samples=json.loads((run/'pipeline_samples.json').read_text())
assert len(samples)==20 and all(s['max_edits']==1 for s in samples)
print('RESULTS_QA_PASS: dataset hashes, development selection, held-out metrics, bounds, official TSAR agreement, no SWORDS context overlap, pipeline samples')
