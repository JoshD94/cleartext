# Default pipeline on Chhoeng's sentence sets, November 23, 2026

Inputs: `passive/ps_sentences.json` (100 political-science sentences) and
`passive/b_sentences.json` (100 business sentences) from Chhoeng's
`passive-to-active` branch at 9b71f43 (arXiv/NBER abstracts; SHA-256
`f45f7812…` and `85297836…`). Each sentence ran alone as a one-sentence passage.
Aggregates: [summary.json](summary.json). Raw text and outputs stay local
(`~/.local/state/cleartext/teammate-sentences-20261123/rows.jsonl`, Yoga).

```sh
python -m evals.sentence_set run --state <dir> ps_sentences.json b_sentences.json   # bounded, resumable
python -m evals.sentence_set summary --state <dir> --out summary.json ps_sentences.json b_sentences.json
```

Systems: the default pipeline (F2 → structure v4 → learned words p50, through
`demo-web/pipeline.DemoPipeline`), and Chhoeng's `passive_to_active` through
`cleartext.teammate_stages.PassiveStage` with placeholder `"auto"` and `None`
(by-phrase passives only), unchanged. NLI is bidirectional DeBERTa-v3-base
MNLI (`evals.preservation.nli_entailment`) on changed sentences only.

| System | Changed / 200 | FK on changed | NLI mean (changed) | NLI < 0.80 |
|---|---:|---|---:|---:|
| Default pipeline | 60 | 17.06 → 13.93 | 0.985 | 0 |
| Passive, "auto" | 20 | 16.25 → 16.64 | 0.871 | 4 |
| Passive, by-only | 3 | 23.13 → 22.83 | 0.666 | 1 |

Default by stage: F2 changed 39 sentences, structure v4 1 (a passive), words
p50 27 (one swap each). No numbers, negations or names were lost. 0 NLI
abstentions, 0 converter errors.

One reading by Claude (not a blind rating): of F2's 39 splits, 25 are clean,
10 leave a sentence fragment ("Particularly those that …"), 3 shift meaning
(wrong referent or a dropped "because") and 1 only moves a quote mark. The 27
word swaps read well (one slight shift: assesses → measures). Of the passive
converter's 20 rewrites, 7 read well, 6 are awkward ("someone"/"people"
subjects) and 7 are garbled; NLI scores most garbled ones above 0.95.

Nothing was tuned or chosen on these sentences; the locked test split was not
read. If they are used to design an F2 fragment check, they become development
data. Queue jobs (Yoga, `later`): `19e8bf36a5fa` (118 sentences, 30 min
budget) and `9850a060428e` (82), about 14 s per sentence, peak RAM 3.9 GB.
