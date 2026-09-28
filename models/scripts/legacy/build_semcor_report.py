import json,html
from pathlib import Path
root=Path(__file__).resolve().parents[2];run=root/'runs/semcor-20260924'
m=json.loads((run/'metrics.json').read_text());e=json.loads((run/'evaluation.json').read_text())
esc=lambda s:html.escape(str(s))
def table(headers,rows):
    return '<div class="wrap"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ClearText: supervised word senses</title><style>
body{overflow-wrap:anywhere;font:16px/1.55 system-ui,sans-serif;color:#172332;background:#f8fafc;margin:0}main{max-width:1080px;margin:auto;padding:32px 20px}h1{font-size:30px;margin-bottom:8px}h2{font-size:21px;margin-top:32px}p{max-width:850px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dbe3ec;vertical-align:top}th{background:#eaf0f6}details{background:white;border:1px solid #dbe3ec;border-radius:6px;margin:10px 0;padding:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.note{padding:16px;border-left:4px solid #b77017;background:#fff6e6}.wrap{overflow-x:auto}a{color:#245eb6}@media(max-width:600px){main{padding:20px 12px}td,th{padding:8px}table{font-size:12px}}
</style><main><p>CS 4701 · 24 September 2026</p><h1>Learning word meanings from SemCor</h1>
<p>Trained on existing sense annotations. No new human ratings, transformer model, or generative LLM.</p>
<p class="note"><strong>Better precision, fewer useful edits.</strong> Sense filtering improves the share of accepted replacements that match human references. It rejects enough good candidates that total correct replacements fall. The new pipeline remains experimental; the basic model is preserved.</p>
<h2>What was trained</h2><p>Each target lemma and part of speech has a model of its possible meanings. Training records supply the correct WordNet sense. The model learns meaning frequencies, surrounding-word counts, and the average context vector for each meaning.</p>
<p>Score = log smoothed training frequency + 3 × mean naive-Bayes context log-likelihood + 5 × cosine similarity to the learned sense vector. These weights were selected on development documents. The target word is excluded from context features.</p>
<p>Context features include sentence content words, nearby words, and words at positions −2, −1, +1, +2. Static vectors are from spaCy en_core_web_md 3.8.0. They are not contextual token embeddings. A trained sense filters WordNet candidates; the original difficulty scorer then ranks simpler candidates. Existing preservation checks remain in the combined version.</p>
<h2>Word-sense evaluation</h2>'''
page+=table(['Measure','Training-frequency baseline','Learned context model'],[
 ['Overall held-out accuracy',f'{m["baseline_test"]["accuracy"]:.1%}',f'{m["test"]["accuracy"]:.1%}'],
 ['Ambiguous words only',f'{m["baseline_test"]["ambiguous_accuracy"]:.1%}',f'{m["test"]["ambiguous_accuracy"]:.1%}']])
page+=f'<p>{m["counts"]["train"]:,} training labels; {m["counts"]["dev"]:,} development labels; {m["counts"]["test"]:,} test labels. Documents are disjoint and exact cross-split sentence duplicates are removed. Multiword targets, named-entity chunks, and incompatible sense mappings are excluded.</p>'
ci=m['accuracy_gain_document_bootstrap_95ci']
page+=f'<p>Gain: {(m["test"]["accuracy"]-m["baseline_test"]["accuracy"])*100:.2f} percentage points. Document-bootstrap 95% interval: {ci[0]*100:.2f} to {ci[1]*100:.2f} points. This custom SemCor split supplies the correct lemma and part of speech; real inputs use predicted parses and can perform worse.</p>'
page+='<h2>Does it improve simplification?</h2>'
page+=table(['TSAR variant','Edited cases','Reference matches','Matches among edits','Matches across all 373'],[
 [r['name'],r['edits'],r['correct'],f'{r["match_among_edits"]:.1%}',f'{r["top1"]:.1%}'] for r in e['tsar']])
page+='<p>TSAR supplies the target word, so these numbers do not measure difficult-word detection. Exact reference matches may miss valid alternatives. Official TSAR evaluator output is saved with the run. Keeping everything yields zero reference matches.</p>'
page+=table(['SWORDS candidate filter','Accepted pairs','Precision','Recall'],[[r['name'],r['accepted'],f'{r["precision"]:.1%}',f'{r["recall"]:.1%}'] for r in e['swords']])
page+='<p>SWORDS uses 45,705 candidate pairs and existing human votes. Positive means at least half of judgments are TRUE. These custom membership metrics measure context suitability, not simplification quality. Test sets were reused from earlier experiments, but neither trained nor selected this SemCor model. Results are exploratory.</p>'
page+='<h2>20 actual word-replacement outputs</h2><p>The same 20 sentences used for the basic model. Phrase and structure changes are disabled here to isolate the new word module. Unchanged does not imply simple or correctly understood.</p>'
page+=table(['#','Original','Basic model','Sense model + checks'],[[i,r['original'],r['baseline'] if r['baseline']!=r['original'] else 'Unchanged',r['output'] if r['output']!=r['original'] else 'Unchanged'] for i,r in enumerate(e['samples'],1)])
page+='<p class="note">Known failures remain: “alleviates pain” becomes “facilitates pain”; “stipulates” becomes “qualifies”. The bank-money probe still selects the river-edge sense. These examples are reported unchanged from the model output.</p>'
page+='<h2>Combined with phrase and structure modules</h2>'+table(['Original','Combined output'],[[r['original'],r['output']] for r in e['composed']])
page+='<h2>Inspect model decisions</h2>'
for i,r in enumerate(e['samples'],1):
    page+='<details><summary>'+str(i)+'. '+esc(r['original'])+'</summary><pre>'+esc(json.dumps(r['trace'],indent=2))+'</pre></details>'
page+='<details><summary>Bank sense probes</summary><pre>'+esc(json.dumps(e['banks'],indent=2))+'</pre></details>'
page+='''<h2>Sources and reproduction</h2><p><a href="https://www.nltk.org/howto/corpus.html">NLTK SemCor</a> · <a href="https://wordnet.princeton.edu/">WordNet</a> · <a href="https://github.com/p-lambda/swords">SWORDS</a> · <a href="https://github.com/LaSTUS-TALN-UPF/TSAR-2022-Shared-Task">TSAR</a> · <a href="https://spacy.io/models/en#en_core_web_md">spaCy vectors</a></p>
<p>Run scripts/train_semcor.py, scripts/evaluate_semcor_pipeline.py, scripts/verify_semcor.py, and scripts/build_semcor_report.py with .venv/bin/python. Inference: scripts/predict_sense.py TEXT; add --full to enable the earlier phrase and structure modules. Artifacts: runs/semcor-20260924.</p></main></html>'''
(root/'outputs/semcor-model-review.html').write_text(page)
print('REPORT_COMPLETE')
