import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cleartext.pipeline import ClearText
from cleartext.data import ROOT
run=ROOT/'runs/initial-20260924'
rows=json.loads((run/'lexical.json').read_text())['review']
model=ClearText()
results=[{'id':r['id'],'domain':r['domain'],**model.analyze(r['original'])} for r in rows]
(run/'pipeline_samples.json').write_text(json.dumps(results,indent=2))
print('PIPELINE_SAMPLES',len(results))
