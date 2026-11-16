# Tunable parameters of the non-LLM ensemble (inventory, November 16, 2026)

Every setting below can change an output of `EnsembleClearText.analyze()` or
one of its stages. "Exposed" says how the sweep harness
([`evals/param_sweeps.py`](../../param_sweeps.py)) sets it. Settings that were
hard-coded are now constructor or keyword arguments whose defaults are the old
values, so no default output changes (checked: the harness reproduces the
November 10 winner and references cell for cell; see the README).
Sweep key = the `PARAMS` name in the harness; "—" = listed but not swept, with
the reason.

## Structure: T5 splitter (the stage that carries the ensemble)

| Param | Location | Current | Plausible range | Expected effect | Sweep key |
|---|---|---|---|---|---|
| Beam width | `structure/seq2seq.py` `SentenceGenerator(num_beams=)` (was hard-coded) | 2 | 1–4 | Wider beams: slightly more fluent splits, a few more passing checks; 1 = greedy | `t5.beams` |
| Beams tried per sentence | `structure/tuned.py` `TunedSplitter(candidates=)`, with `num_return_sequences` (new, opt-in) | 1 (top beam only) | 1–4 | Try lower beams when the top one fails a check: more splits without loosening a check | `t5.candidates` |
| Max new tokens | `SentenceGenerator(max_new_tokens=)` (was hard-coded) | 256 | 128–256 | Never binds (longest output < 200 tokens) | — |
| No-repeat n-gram | `SentenceGenerator(no_repeat_ngram_size=)` (was hard-coded) | 0 | 0, 3 | Would block a copied subject from being repeated, which splits need | — (harmful by design) |
| Source token limit | `SentenceGenerator(max_source_tokens=)` (was hard-coded) | 192 | 128–256 | Never binds on any panel (v2: longest 134 tokens) | — |
| Per-sentence checks (16 soft + 5 hard) | `structure/seq2seq.py:generation_checks` → `common.structure_checks` + discourse, tense, complement scope, termination | all must pass | waive any soft one | Each waived check admits more splits (FK ↓) at some meaning risk | `t5.waive` |
| Hard checks: numbers, negation, entities, code literals, technical phrases | same | always on | never waived | Guard frozen gates 3–4 | — (only to replay `t5_raw`/`t5_unchecked`) |
| Passage-level check | `structure/adapter.py:rewrite` → `structure_checks` on the whole passage (sentence count 1–4, no fragment anywhere) | `single` | `single`, `passage` (compound's passage-aware check, now `compound.passage_checks`), `off` | `single` rejects any passage with > 4 output sentences or an untouched fragment elsewhere; `passage` fixes both | `t5.passage` |
| Minimum sentence length to split | `TunedSplitter(min_words=)` (new) | 0 | 0–20 words | Skip short sentences: fewer tiny sentences, fewer risky splits | `t5.min_words` |
| Per-split NLI acceptance | `TunedSplitter(entailment=, min_entailment=)` (new); scorer = Chhoeng's `phrase_replacer.entailment` (cross-encoder nli-deberta-v3-small, min of both directions), not the evaluation model | off | 0.5–0.95 | A meaning floor that can replace structural checks: loosen checks, keep splits the NLI accepts | `t5.min_entailment` |

## Structure: compound splitter (PR #12 + tagger)

| Param | Location | Current | Plausible range | Expected effect | Sweep key |
|---|---|---|---|---|---|
| Tagger threshold | `structure/compound.py` `DependencyCompoundSplitter(tagger_threshold=)` (was the artifact's 0.4) | 0.4 | 0.2–0.7 (tuning curve: precision 0.88→1.0, recall 0.84→0.04) | Lower: more tagged boundaries, more splits | `cmp.threshold` |
| Minimum sentence words | `DependencyCompoundSplitter(min_sentence_words=)` | 0 | 0–15 | Fewer splits of short sentences | `cmp.min_sentence_words` |
| Minimum clause words | `DependencyCompoundSplitter(min_clause_words=)` | 0 | 0–5 | No very short halves | `cmp.min_clause_words` |
| Guards | `DependencyCompoundSplitter(guards=)` | `scoped` | `scoped`, `strict` | `strict` = Nov 2 behavior; the tagger requires `scoped` | — (strict excludes the tagger) |
| Sentence shape limit | `compound.py` `_sentence`: one sentence, ends with ".", ≤ 120 tokens | fixed | — | Abstains on other shapes | — (shape contract) |

## Phrases

| Param | Location | Current | Plausible range | Expected effect | Sweep key |
|---|---|---|---|---|---|
| Authored rules: min complexity gain | `rewrites.py:phrase_rewrite(min_gain=)` (was hard-coded) | 0.02 | 0–0.1 | Lower admits more authored edits | `phr.min_gain` |
| Authored rules: edits per passage | `phrase_rewrite(max_edits=)` (was hard-coded "one phrase edit") | 1 | 1–3 | More authored edits per passage | `phr.max_edits` |
| Phrase-finder sources | `teammate_stages.PhraseFinderStage(sources=)` | per level (`crf` = manual, verified, model) | subsets | Which of Chhoeng's lists fire | `finder.sources` |
| CRF span threshold | `pos_tag/phrase_finder.py:171` `MODEL_THRESHOLD`; set per call via `PhraseFinderStage(settings=)` (new) | 0.3 | 0.2–0.7 | Higher: fewer, surer model spans | `finder.model_threshold` |
| Replacer NLI floor | `pos_tag/replacer_settings.json` `min_entailment` → `phrase_replacer.MIN_ENTAILMENT` | 0.575 | 0.5–0.95 | Higher: fewer model replacements | `finder.min_entailment` |
| Replacer similarity floor | `phrase_replacer.py:58` `MIN_SIMILARITY` | 0.80 | 0.7–0.95 | Higher: fewer replacements | `finder.min_similarity` (not swept; same role as the NLI floor) |
| Replacer fluency ceiling | `replacer_settings.json` `max_fluency_drop` | 0.966 | 0.5–1.0 | Lower: fewer replacements | `finder.max_fluency_drop` (not swept; same role) |
| SimplePPDB score cut-off | `simpleppdb_phrases.csv` `ppdb` column (4.0–5.39; median 4.15) — the loader ignored it; `settings={"ppdb_min":}` filters matches (new) | none | 4.3–5.0 | Keep only high-confidence paraphrases | `finder.ppdb_min` |
| BERT top-k, agree-top, max words, frequency margin, tries, fluency window | `phrase_replacer.py:57,74–81` | 20, 10, 2, 0.5, 1, 3 | — | Candidate generation inside his replacer | — (his tuned search; the floors above gate its output) |

## Passive (Chhoeng, PR #14)

| Param | Location | Current | Plausible range | Expected effect | Sweep key |
|---|---|---|---|---|---|
| Mode / placeholder | `teammate_stages.PassiveStage(placeholder=)` → `passive_to_active.PassiveToActive` | off (`on` = `auto`, `agentive` = by-phrase only) | `auto`, `someone`, `people`, agentive | Agentless passives get a placeholder subject; fixed placeholders are blunter than `auto` | factor level + `passive.placeholder` |

## Words (v7 and contextual runs)

| Param | Location | Current | Plausible range | Expected effect | Sweep key |
|---|---|---|---|---|---|
| Decision acceptance threshold | run `config.json` `decision_threshold` → `EnsembleClearText.decision_threshold` | 0.45 | 0.45–0.9 | Higher: high-confidence swaps only | `w.threshold` |
| CWI detector threshold | `runs/ensemble-v5-20260927/detector.pkl` `threshold` | 0.39 | 0.39–0.7 | Higher: only clearly hard words are targeted | `w.detector` |
| Noun hardness threshold | `config.json` `noun_threshold` | 0.8 | 0.8–0.95 | Higher: fewer noun targets | `w.noun` |
| Meaning floor (sense fit) | `config.json` `meaning_floor` | 0.1 | 0.1–0.5 | Higher: drop candidates unlikely to share the target's sense | `w.floor` |
| Word edits per passage | `config.json` `max_word_edits` | 1 | 1–2 | More swaps; each is a meaning risk | `w.max_edits` |
| Hardness exponent | `config.json` `hardness_alpha` | 0.5 | 0–2 | Which flagged word gets the one edit | `w.alpha` |
| Candidate difficulty-gain margin | `EnsembleClearText(word_min_gain=)` (new) | none | 0.1–0.2 | Only clearly simpler candidates | `w.min_gain` |
| Candidate frequency margin | `EnsembleClearText(word_zipf_margin=)` (new) | none | 0.5–1.0 zipf | Only clearly more common candidates | `w.zipf_margin` |
| Entailment veto threshold | `teammate_stages.EntailmentVeto(threshold=)` | 0.575 | 0.575–0.97 | Stricter meaning floor per swap | `w.veto` |
| Veto refill | `EnsembleClearText(veto_refill=)` (new; was always on) | true | true/false | false: a vetoed slot is not handed to the next-ranked word (v2's failure mode) | `w.veto_refill` |
| Error cost, difficulty cutoff, tiering | `config.json` | 0.05, 0.3, false | — | Used only without a decision model or detector; v7/contextual have both | — (inactive) |
| Sense/fit model internals (temperatures, weights, BERT sense weight) | `config.json` | trained | — | Trained model settings; retuning them is retraining | — |

## Grammar, negation, terms

| Param | Location | Current | Plausible range | Expected effect | Sweep key |
|---|---|---|---|---|---|
| Negation rewrite | `rewrites.negation_rewrite`, curated map | off | on/off | Fires only on double negatives / "not un-" | factor level (on/off only; the map is curated, not a threshold) |
| Grammar corrector edits | `grammar_preprocessing.GrammarCorrector(max_edits=, timeout=)` | 8, 5 s | 0–8 | LanguageTool fixes before simplifying | — (needs a LanguageTool server; v1/v2 showed identical output) |
| Technical-term protection | `EnsembleClearText.technical_terms` | off | on/off | Blocks stages that change a detected term | — (identical output in v2) |
| Guardrails | `analyze(guardrails=)` | on | on | Lexical/literal safety | — (never off) |
