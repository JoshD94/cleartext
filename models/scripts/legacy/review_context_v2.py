import sys,json,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cleartext.context_pipeline import ContextClearText
from cleartext.features import nlp
from cleartext.semantics import sense_scores
from cleartext.data import ROOT
run=ROOT/'runs/context-20260924'
examples=[
 ('word','The software utilizes a database to store customer records.'),
 ('word','Please terminate the program before restarting your computer.'),
 ('word','The server requires authentication before granting access.'),
 ('word','The medication alleviates pain but can cause drowsiness.'),
 ('context','She deposited money at the bank.'),
 ('context','They sat on the bank beside the river.'),
 ('context','The contract stipulates that payment is due within thirty days.'),
 ('phrase','Despite the fact that it rained, the match continued.'),
 ('phrase','The office is closed at the present time.'),
 ('phrase','All employees attended with the exception of Maria.'),
 ('phrase','A large number of students submitted the assignment.'),
 ('phrase','We check the server on a daily basis.'),
 ('phrase','The clinic is in close proximity to the station.'),
 ('phrase','Save your work prior to restarting the computer.'),
 ('phrase','In the event that the alarm sounds, leave the building.'),
 ('phrase','She studies in order to pass the exam.'),
 ('structure','The report was published by the committee.'),
 ('structure','The engineer tested the software, and the manager reviewed the report.'),
 ('combined','The report was published by the committee, and a large number of students read it.'),
 ('keep','If the server fails, the backup may start.')
]
model=ContextClearText();rows=[]
for i,(kind,text) in enumerate(examples,1):
    output=model.analyze(text)
    old=model.base.analyze(text,mode='learned')['output']
    rows.append({'id':i,'kind':kind,'baseline':old,**output})
    print(i,text,'=>',output['output'],flush=True)
probes=[]
for text in [examples[4][1],examples[5][1]]:
    doc=nlp()(text);token=next(t for t in doc if t.text=='bank')
    probes.append({'text':text,'target':'bank','scores':sense_scores(doc,token)})
old20=json.loads((ROOT/'runs/initial-20260924/basic_examples_20.json').read_text())
comparison=[{'baseline':r['output'],**model.analyze(r['original'])} for r in old20]
result={'examples':rows,'previous20':comparison,'sense_probes':probes,
        'context_metrics':json.loads((run/'metrics.json').read_text()),
        'phrase_metrics':json.loads((run/'phrase_metrics.json').read_text()),
        'note':'Authored demonstrations, not independent accuracy measurements. Prior word examples are regression checks. SWORDS test was previously inspected; reused benchmark results are exploratory.'}
(run/'review.json').write_text(json.dumps(result,indent=2))
e=lambda x:html.escape(str(x))
table=''.join('<tr><td>'+str(r['id'])+'</td><td>'+e(r['original'])+'</td><td>'+e(r['output'] if r['output']!=r['original'] else 'Unchanged')+'</td><td>'+e(', '.join(x['stage'] for x in r['edits']) or 'keep')+'</td></tr>' for r in rows)
details=''.join('<details><summary>'+str(r['id'])+'. '+e(r['original'])+'</summary><p>Basic model: '+e(r['baseline'])+'</p><pre>'+e(json.dumps({'edits':r['edits'],'trace':r['trace']},indent=2))+'</pre></details>' for r in rows)
metrics=result['context_metrics'];pm=result['phrase_metrics']
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ClearText: context and phrase experiment</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#172332;background:#f8fafc;margin:0}main{max-width:1080px;margin:auto;padding:32px 20px}h1{font-size:30px;margin-bottom:8px}h2{font-size:21px;margin-top:32px}p{max-width:850px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dbe3ec;vertical-align:top}th{background:#eaf0f6}details{background:white;border:1px solid #dbe3ec;border-radius:6px;margin:10px 0;padding:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.note{padding:16px;border-left:4px solid #b77017;background:#fff6e6}.wrap{overflow-x:auto}a{color:#245eb6}@media(max-width:600px){main{padding:20px 12px}td,th{padding:8px}table{font-size:12px}}
</style><main><p>CS 4701 · 24 September 2026</p><h1>Context, phrases, and sentence structure</h1>
<p>Experimental addition to the frozen initial model. Understand original context → apply a syntax rule → simplify a phrase → consider a word replacement.</p>
<p class="note"><strong>The context checker is still too weak.</strong> Neither trained candidate reached 70% development precision with at least 30 accepted pairs. This conservative configuration therefore disables word substitutions. Keeping words is abstention, not proof that the system understands them.</p>
<h2>What changed</h2><ul><li>Context: 300-dimensional static spaCy vectors averaged across surrounding content words and WordNet definitions. A gradient-boosted classifier learns candidate suitability from SWORDS human ratings.</li><li>Phrase complexity: a gradient-boosted regressor trained on 9,179 CompLex word and phrase examples. Ten authored phrase pairs supply alternatives. This is not a general phrase generator.</li><li>Structure: two parser rules handle explicit-agent passives and independent clauses joined by comma + and. Conditions, negation, modals, questions, and quoted text are excluded.</li><li>Every candidate has a score and rejection reasons. Original sentence context remains available after structural edits.</li></ul>
<h2>Measured results</h2>'''
page+=f'<p>Phrase difficulty average absolute error: <strong>{pm["multiword_test_mae"]:.3f}</strong> on 184 held-out phrases, on the 0–1 scale. Mostly two-word training targets; longer phrases are extrapolations.</p>'
page+=f'<p>Context discrimination, measured by AUC: <strong>0.658 → {metrics["test"]["auc"]:.3f}</strong>. At a 0.5 cutoff, only {metrics["test_at_half"]["precision"]:.1%} of accepted candidate pairs received a positive human label; recall was {metrics["test_at_half"]["recall"]:.1%}. The conservative deployment cutoff accepts none.</p>'
page+='''<p>These SWORDS test data were used in the previous experiment. Selection used development data only, but this remains an exploratory comparison, not a fresh final test.</p>
<h2>Methods and limits</h2><p>Context vector = mean of available content-word vectors, normalized to unit length. Definition vectors include the gloss, usage examples, and hypernym definitions. Cosine similarity is their dot product. The checker combines those similarities with existing dictionary and local-context features.</p>
<p>Phrase score = clip(boosted-tree prediction, 0, 1). Inputs include token count, character count, and mean/min/max word frequency, length, and syllables. The phrase replacement must reduce the predicted score by at least 0.02. The ten candidate pairs are authored; they are not extracted from Simple PPDB.</p>
<p>The bank probe exposes a failure: money context favors a gambling-funds definition over the intended financial institution. River context favors the correct river-edge definition by only a small margin. The model also overestimates difficulty for some function words, so it refuses “in order to → to” and “in the event that → if”.</p>
<h2>20 actual outputs</h2><p>Examples illustrate behavior and were authored for review. They do not establish accuracy. A combined rewrite can still make mistakes.</p><div class="wrap"><table><thead><tr><th>#</th><th>Original</th><th>Output</th><th>Edited stage</th></tr></thead><tbody>'''
page+=table+'</tbody></table></div><h2>Inspect decisions and baseline comparison</h2>'+details
page+='<h2>Sources and reproduction</h2><p><a href="https://github.com/MMU-TDMLab/CompLex">CompLex</a> · <a href="https://github.com/p-lambda/swords">SWORDS</a> · <a href="https://spacy.io/models/en#en_core_web_md">spaCy vectors</a> · <a href="https://wordnet.princeton.edu/">WordNet</a></p><p>Use scripts/train_context_v2.py, scripts/train_phrases.py, and scripts/review_context_v2.py with the project virtual environment. Run artifacts: runs/context-20260924. Data hashes and development selections are recorded there. The original report and model remain available.</p></main></html>'
(ROOT/'outputs/context-model-review.html').write_text(page)
print('REVIEW_COMPLETE',flush=True)
