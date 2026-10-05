# Experiment log

Chronological record of every model version and refinement round, with the numbers that decided each change. Scripts for comparison rounds are in `scripts/experiments/`; scripts that build shipped artifacts are in `scripts/`. TSAR test and SWORDS test were reused across rounds, so their numbers are exploratory. Choices were made on development data only.

## Initial results

The selected word-only model is gradient boosting, test MAE 0.0696 on CompLex. Context features select random forest, test MAE 0.0690. This small difference is not clearly established by the example bootstrap interval. Sentence gradient boosting reaches MAE 0.4573 on the mean of the two CEFR educator ratings.

The context-fit classifier is weak. Its custom SWORDS binary diagnostic has test AUC 0.658 and precision 0.151 at the development-selected operating threshold. Dictionary substitution has official TSAR Precision@1 0.2600; the full guarded variant reaches 0.2064. KEEP scores zero in that metric. The added modules do not yet improve replacement quality.

Preservation checks catch six of eight deliberately incorrect rewrites, but miss an antonym and lost specificity. They accept both controls. Passing checks is not proof of semantic equivalence.

## Evaluation boundaries

Preserve all official splits. Choose configurations using development results, never test results. SWORDS development contexts are split for training/validation; official test contexts are separate. The context classification target is TRUE votes divided by all judgments, thresholded at 0.5. These are custom classification diagnostics, not official SWORDS ranking metrics.

The ranker's difficulty margin 0.02, context weight 0.03, and full pipeline's difficulty cutoff 0.30 are initial heuristic choices. Only the context-fit threshold is development-tuned. TSAR supplies target words, so its table does not evaluate difficult-word detection. The report's authored examples exercise the full detector and one-edit pipeline.

Do not retrain to improve the displayed test scores. Use new development evidence and independent human review for the next iteration. The initial pipeline has no sentence restructuring or phrase replacement. The experiment below adds limited rules. An ordinal CEFR model, comprehensive jargon detection, and general meaning verification are not implemented.

## Artifacts

- `outputs/cleartext-model-review.html`: standalone review page with formulas, comparisons, examples, and exportable ratings.
- `runs/initial-20260924/`: frozen models, selected configurations, predictions, evaluation output, and report data.
- `logs/`: setup, training, and evaluation logs.
- `work/`: browser QA screenshots and temporary verification files.

The HTML works offline. Ratings are stored locally in the reviewer's browser and exported as JSON. They do not automatically retrain models. Review examples contain attributed public dataset text; source licenses are listed in the report. Do not redistribute the datasets under a blanket project license.

## Context and phrase experiment

The separate context_pipeline.py adds static-vector context features, a trained candidate checker, ten authored phrase pairs ranked by a CompLex word/phrase regressor, and two narrow syntax rules. The current checker abstains from all word substitutions. It does not fix word meaning reliably. Phrase and structural examples are demonstrations, not an accuracy estimate. Longer phrases extrapolate beyond the predominantly two-word training targets.

Run scripts/train_context_v2.py, scripts/train_phrases.py, then scripts/review_context_v2.py with .venv/bin/python. Prerequisite: the initial training and lexical evaluation have run. Inference: scripts/predict_context.py TEXT. Tests: pytest tests -q, scripts/verify_context_v2.py, and scripts/verify_context_report.py. The report is outputs/context-model-review.html and experiment artifacts are in runs/context-20260924. The older pipeline and report remain unchanged.

## SemCor sense experiment

The separate SenseClearText pipeline learns word meanings from SemCor and filters WordNet candidates to the chosen sense. It restores word edits, while the v2 conservative checker remains available unchanged. Custom held-out SemCor accuracy is 69.1% versus 67.7% for a training-frequency baseline. On reused TSAR data, reference matches among edited cases rise from 43.4% to 53.6%, but total correct replacements fall from 85 to 52. This is an experimental precision/coverage tradeoff, not an overall win.

Run scripts/train_semcor.py, scripts/evaluate_semcor_pipeline.py, scripts/verify_semcor.py, and scripts/build_semcor_report.py using .venv/bin/python. Inference uses scripts/predict_sense.py TEXT, with --full to add earlier phrase and structure rules. The report is outputs/semcor-model-review.html. Artifacts are in runs/semcor-20260924. SemCor is installed by scripts/setup.sh.

## Ensemble experiment

`ensemble.py` defines three abstract model roles, each with an ensemble that implements the same interface: `DifficultyModel` / `DifficultyEnsemble`, `SenseDistribution` / `SenseEnsemble`, and `FitScorer` / `StackedFit` (or the untrained `AverageFit`). New models join a role by subclassing it; the pipeline in `ensemble_pipeline.py` only sees the role interface.

Sense members: SemCor (refit with tuned prior smoothing) and WordNet gloss vectors. The SemCor weight shrinks to n/(n+3) for a lemma with n training examples, so one SemCor example no longer decides the sense. Fit members: sense fit, round-trip sense, WordNet first sense, candidate sense rank, the v2 SWORDS checker, context vectors, argument vectors, Brown bigrams and WordNet membership. A logistic stacker combines them, trained only on the SWORDS dev contexts the v2 checker never saw. The hard top-1 sense filter, the unseen-lemma rejection and the fixed fit threshold are gone. A candidate is applied when fit*gain > (1-fit)*error_cost; error_cost is chosen on the same SWORDS holdout.

Results, all exploratory on reused test sets. SemCor test: likelihood improves (-0.904 vs -0.945), top-1 accuracy drops from 69.1% to 67.8%. SWORDS test AUC on generated candidates: 0.784, versus 0.762 for the best single member. TSAR test: 87 correct from 180 edits (48.3%), versus 85 of 196 (43.4%) for the dictionary baseline and 52 of 97 (53.6%) for the previous sense filter. At higher error_cost the curve passes both old points, e.g. 79 of 138 (57.2%) at 0.05; that sweep is diagnostic and skips guardrails.

Fixed: alleviates -> relieves (was facilitates), and coverage lost to the old filters. Still wrong at the selected error_cost 0.01: stipulates -> qualifies, exacerbate -> exasperate, deposited -> stuck. The error_cost was selected on only 80 SWORDS dev targets and 26 edits, so the operating point is weakly determined.

WordNet sense order derives from tagged-corpus frequency, largely SemCor. It is therefore used only as a SWORDS-trained fit member, never in the SemCor-tuned sense mixture.

Run scripts/train_ensemble_senses.py, then scripts/train_ensemble_fit.py, with .venv/bin/python. Inference: scripts/predict_ensemble.py TEXT, with --full to add phrase and structure rules. Tests: pytest tests -q. Artifacts are in runs/ensemble-20260927. Older pipelines and artifacts are unchanged; SenseModel's new parameters default to the old behavior.

## Ensemble v2

`generation.py` adds candidate generation as a fourth interchangeable role (`CandidateGenerator`, `WordNetGenerator`, `GeneratorUnion`). v2 proposes WordNet synonyms, hypernyms and verb groups, similar-to and also-see words, and multiword lemmas. When the tagger's part of speech has no WordNet entry, it retries other parts of speech using WordNet's lemmatizer. Each candidate records its source; flag members let the stacker learn each source's risk. Selection is tiered: a hypernym or similar-to word is used only when no synonym has positive utility. SWORDS dev chose tiering over flat ranking (net +12 vs +10 edits). a/an agreement is repaired from CMUdict pronunciations. The synonym-only generator reproduces lexical.generate except for 3 TSAR targets, where the old generator re-inflected the target's own lemma (primed -> primmed).

The v2 checker is cross-fitted in 5 grouped folds, so the stacker and error_cost now use all 370 SWORDS dev contexts (353 targets with generated candidates, up from 80). The stage-1 sense ensemble is reused unchanged. Selected error_cost: 0.05.

TSAR test, reused and exploratory: 119 correct from 223 edits (53.4%), versus 87 of 180 (48.3%) for v1 and 52 of 97 (53.6%) for the old sense filter. A correct answer is now among the candidates for 310 of 373 targets (196 before), and only 10 targets have no candidates (84 before). By source: synonyms 65 of 118 correct, hypernyms 42 of 91, similar-to 12 of 14. TSAR rewards general words; hypernyms can still lose specificity that no guardrail checks (encodes -> converts, deposited -> gave). SWORDS test, generated candidates: AUC 0.768, 54 of 108 edits acceptable at the operating point.

Run scripts/train_ensemble_senses.py (stage 1), then scripts/train_ensemble_v2.py. scripts/predict_ensemble.py now loads v2; EnsembleClearText.load(RUN_V1) loads v1. Artifacts are in runs/ensemble-v2-20260927.

## Ensemble v3 and refinement rounds

Two datasets were added. BenchLS (Paetzold and Specia 2016, CC BY 4.0, 929 sentences with ranked simpler substitutes) is split by target word into dev (424) and a fresh holdout (505). CWI 2018 English (News, WikiNews, Wikipedia; organizers' Sheffield NLP mirror, pinned commit, no license file, not redistributed) has 23,562 single-word train rows, 2,884 dev and 3,701 test. Every choice below was made on BenchLS dev or CWI dev; the BenchLS holdout, CWI test and TSAR test were read only for reporting. Scripts: build_candidate_tables.py caches candidate scores; experiments/refine_round1_2.py, refine_round3.py to refine_round5.py, train_detector.py and refine_target_choice.py run the rounds; results are in runs/ensemble-v2-20260927/refine_round*.json and runs/ensemble-v3-20260927/.

| Round | Change | Dev | Kept? |
|---|---|---|---|
| 1 | Decision model trained on BenchLS dev over the SWORDS-trained members | net 29 -> 51 | yes |
| 2 | Frequency, length and syllable differences; gain is a feature, not a gate | 51 -> 67 | yes |
| 2 | Relax the argument_roles guard | no gain | no |
| 3 | Listwise features (standing among the case's candidates), shared prefix, spelling variant | 67 -> 69 | yes |
| 4 | Gradient boosting or averaged decision model | 70 -> 48 / 63 (10-fold x3) | no |
| 5 | Candidate already in the sentence, as feature or rule | no gain / hurts | no |
| D | CWI-trained target detector (boosting over both CompLex scores + word features) | CWI F1 0.43 -> 0.78 | yes |
| T | Choose the edited word by detector or product instead of decision probability | 75 -> 63 / 69 | no |

Held-out results for the dev-only decision model: BenchLS holdout 193 of 341 edits correct (56.6%), versus 114 of 219 (52.1%) for v2. TSAR test 116 of 218 (53.2%), versus 119 of 223 for v2. The shipped decision.pkl is refit on all BenchLS after those numbers were recorded; TSAR 125 of 230 (54.3%). Detection on CWI test: F1 0.784 versus 0.378 for the old 0.3 cutoff, which missed three of four words annotators marked.

Round 4 note: boosting and the averaged model scored better on BenchLS holdout and TSAR but worse on dev under every CV scheme; the dev choice was kept. Function words are no longer simplification targets; CompLex never rated them. max_word_edits (CLI --max-edits) allows several word edits per sentence; editing every accepted word roughly doubles correct fixes on the benchmark target in end-to-end runs but adds about 2.9 edits per sentence that these labels cannot check, and interactions between edits are not checked. Default stays 1.

scripts/train_ensemble_v3.py builds the run; scripts/predict_ensemble.py now loads v3 (LATEST).

## Ensemble v4

Round 6 added WordNet specificity: how many senses are filed under the candidate's sense versus the target's (log ratio), and the candidate sense's depth. Its learned weight is negative; broader replacements are accepted less. On the 424-case BenchLS dev the result was mixed and noisy (repeats 60 to 71), while both test sets improved; the round-4 model swap showed the same pattern. Round 7 therefore made all 929 BenchLS cases the dev set (10-fold x3 grouped by target word) and added out-of-fold log-loss as a threshold-free dev metric. From here TSAR test is the only report-only set, and BenchLS no longer has an untouched holdout.

| Decision model (round 7) | Dev net, mean of 3 | Dev log-loss | TSAR test |
|---|---|---|---|
| Logistic (v3) | 142.7 | 0.2695 | 106/188 |
| Logistic + specificity | 134.3 | 0.2686 | 111/183 |
| Average of logistic and boosting + specificity (v4) | 148.0 | 0.2667 | 114/181 |

Both dev metrics pick the average with specificity; specificity lowers log-loss for every model. v4 (scripts/train_ensemble_v4.py, runs/ensemble-v4-20260927, threshold 0.45) is the CLI default. TSAR test end to end: 114 correct of 180 edits (63.3%), versus 125 of 230 (54.3%) for v3 and 119 of 223 (53.4%) for v2. The pipeline reproduces the table evaluation exactly. Fragments of hyphenated words ("re" in "re-enacting") are no longer targets.

## Ensemble v5 (current default) and rounds 8 to 12

Adoption rule from round 11 on: a change must beat run-to-run noise on dev, meaning mean net +5 (about the spread between CV repeats) or out-of-fold log-loss -0.001. Every earlier adopted change passes it.

| Round | Block | Change | Dev result | Kept? |
|---|---|---|---|---|
| 8 | detector, sentence | Detector gain and reading-level change as decision features | net 147 -> 143/146, log-loss equal | no |
| 8 | safety | Antonym check | never fires: generated candidates are never WordNet antonyms | no |
| 9 | context | Checker and fit stacker trained on SWORDS dev + test | fit AUC 0.769 -> 0.779; decision net 147 -> 150, log-loss 0.2667 -> 0.2649 | yes |
| D2 | detector | Tune boosting settings; refit on CWI train + dev | original settings already best; test F1 0.784 -> 0.779 (noise) | refit kept |
| S | sentence | Detector summaries as CEFR features | MAE dev 0.460 -> 0.431, test 0.457 -> 0.432 | yes |
| G | safety | Reject edits that raise the predicted reading level | net 143 -> 104 to 139 | no |
| W | word choice | Choose the edited word by decision probability x hardness^0.5; nouns need hardness >= 0.8 | BenchLS target fixes 161 -> 164, CWI dev net -105 -> -72 | yes |
| 10 | decision | Target part of speech as features | net 150 -> 162.7, log-loss 0.2649 -> 0.2627 | yes |
| 11 | meaning | SemCor refit on train + dev (v6) | net +1, log-loss -0.0002; SemCor test 69.18% -> 69.20% | no (below noise) |
| 12 | decision | Tune logistic C, boosting leaves, mixing weight | best +1.3 net or -0.0009 log-loss | no |
| C | context | Tune checker boosting settings | AUC +0.002 | no |
| M | pipeline | Two word edits per sentence | BenchLS target fixes 164 -> 246, CWI hard share 0.31 -> 0.29, but second edits read worse in samples | option only |

v5 on TSAR test (report only): 152 correct of 239 edits (63.6%), versus 114 of 180 for v4 and 125 of 230 for v3. Word choice on CWI test (report only): share of edits on words most annotators marked hard 0.24 -> 0.32. The sentence block is now wired into the pipeline and reports reading_level_before/after. Safety checks run lazily (identical selections; TSAR in 29 s instead of minutes). v6 is kept on disk but is not the default.

No labelled data exists here for the phrases and structure block, so it was not changed. The word difficulty block was not refit: its only downstream uses are features the decision model already weights, and round 8 showed further difficulty-style features add nothing.

## Context fixes after the "sentence" / "convoluted" review (September 28)

Review case: "Although this sentence is convoluted, the main idea is simple." The meaning model already favored the right sense of "sentence" (68%), but that sense has no WordNet synonyms, so every candidate came from the prison senses. "convoluted" was tagged as a passive verb, so only verb senses ("turned", "twisted") were offered.

| Change | Result | Kept? |
|---|---|---|
| Meaning floor: reject candidates with meaning match < 0.10 | BenchLS dev net 162.7 -> 168.7 (all three CV repeats up); TSAR 152/239 -> 152/238 | yes (meaning_floor in config) |
| Higher floors (0.15 to 0.40) | no further gain; 0.40 hurts | no |
| Adjective reading for every agentless "be + participle" | dev net -4 (noise); read real passives ("is diverted") as adjectives | no |
| Adjective reading only when WordNet counts favor the adjective or the base verb is rare (zipf < 2.5) | changes 1 of 929 BenchLS targets; TSAR identical (152/238); "were convoluted" -> "complex", "was assassinated" still -> "killed" | yes |

The floor at 0.10 does not block "prison term" (meaning match 0.14); at default strictness the decision model already rejects it. BERT side test (scripts/experiments/bert_*.py, separate .venv-bert): combined with our sense ensemble, held-out SemCor sense accuracy rises from 62.9% to 66.9% (mix weight tuned on the other half of a 1,500-annotation sample); BERT alone 58.7%.
BERT fit benchmark (bert_benchmark.py part A, bert_compare.py): BERT-base slot log-probability for all 29,462 SWORDS dev + generated test pairs, 2.5 hours on 4 CPU threads. Alone it reaches AUC 0.666 on generated pairs; as two extra stacker columns (slot score and score relative to the original word) the cross-validated fit rises from AUC 0.779 to 0.800 and log-loss from 0.388 to 0.374, well past the noise bar. Not added to the pipeline yet: it needs the team's decision on using a pretrained encoder, and about half a second to a second per hard word on CPU.

## Ensemble v7: BERT as a meaning and fit member (September 28)

BERT-base (uncased, 110M parameters, CPU) added as two members; everything else unchanged. `cleartext/contextual.py` loads it; `scripts/train_bert_senses.py`, `scripts/train_fit_v7.py`, `scripts/train_ensemble_v7.py` build it.

| Stage | v5 | v7 | Data |
|---|---|---|---|
| Meaning (mix weight 0.2 chosen on SemCor dev by log-likelihood) | 61.8% | 67.6% (BERT alone 65.0%) | 3,000 fresh SemCor test words |
| Fit stacker (+ BERT meaning; + BERT slot score and slot score relative to the original) | AUC 0.779 | 0.791; 0.808 | SWORDS dev + test, grouped 5-fold CV |
| Decision (same model, meaning floor 0.10) | net 168.7, log-loss 0.2627 | net 243.7, log-loss 0.2405 | BenchLS, 10-fold x3 |
| TSAR test (report only) | 152 / 238 (63.9%) | 158 / 219 (72.1%) | 373 sentences |

Sense vectors: mean BERT vector (last four layers) of every SemCor train example for 19,102 senses; the other 98,557 WordNet senses use the lemma read inside its own definition. The round-trip member keeps the SemCor/gloss ensemble (a BERT pass per candidate would be too slow). Caveat: BERT was pretrained on Wikipedia and books, and BenchLS sentences come from Wikipedia, so BERT may have seen those sentences (not their labels) during pretraining; TSAR's gain is smaller but in the same direction.

## Subsequent work, October 2 to 3

These experiments keep v7 as the library and demo default. All BenchLS cases are now reused development data, including the historical holdout. SWORDS historical dev and test were used in fitting; TSAR remains reused reporting data. Scores below do not establish independent held-out accuracy or sentence-level meaning preservation.

| Change | Result and deployment status | Detailed report |
| --- | --- | --- |
| Attributed domain audit and baselines | Fixed ungraded inputs with sources, candidate reasons, identity and dictionary baselines | [Domain audit](DOMAIN_AUDIT.md) |
| Nine relation features | Mean net gain 27.33 at a fixed threshold over v7, saved as an opt-in 48-feature ranker | [Relation decision](RELATION_DECISION.md) |
| Target-centered BERT meaning windows | Retains targets beyond the old cutoff; short BenchLS inputs do not measure the repair | [Context window](CONTEXT_WINDOW.md) |
| Noun phrase-head inflection | Repairs plural forms, but decision refitting and frozen comparisons missed the gain requirement | [Noun inflection](NOUN_INFLECTION.md) |
| Complete BERT slot scoring | Retains every candidate piece with bounded batches; no selection gain in the comparison | [Complete slot](COMPLETE_SLOT.md) |
| Three BERT proposal-support features | Mean net gain 110.33 over the relation ranker; candidates remain WordNet proposals | [Prompted relations](PROMPTED_RELATIONS.md) |
| Full-span replacement similarity | Mean net gain nine, log-loss decrease 0.00818; saved opt-in 52-feature ranker | [Context similarity](CONTEXT_SIMILARITY.md) |
| Detail evidence and technical terms | Learned detail extension rejected; rule-based term protection and graph reports retained as opt-in blocks | [Detail preservation](DETAIL_PRESERVATION.md) |
| All six inspection blocks | Grammar, explanations, consistency and quality reports added; final grammar ablation had zero selection change | [Building blocks](BUILDING_BLOCKS.md) |
| Real-text diagnostic extension | Twelve attributed Europarl sentences from reused CompLex component data, with no correctness labels | [Parliament audit](PARLIAMENT_AUDIT.md) |
| Replacement-sense contrasts | Two variants missed the development acceptance bar; no promotion | [Sense contrast](SENSE_CONTRAST.md) |
| Novel candidate validation | Dictionary and provenance checks distinguish proposal coverage from accepted edits | [Candidate limits](CANDIDATE_LIMITS.md) |
| Novel context validators | Latest full-span similarity shadow gains 14.67 net but loses 3.34 precision points; no calibrated threshold qualifies, so safe novel edits stay disabled | [Novel context](NOVEL_CONTEXT.md) |

Each report retains its original protocol, comparison, failures, saved output paths and source fingerprints. Test counts in those reports describe the suite at that stage. The current suite is documented in the PR validation.
