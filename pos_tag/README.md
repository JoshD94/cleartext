# Sentence Analysis for Text Simplification

Part of our AI sentence simplification tool. This module analyzes a sentence and finds what can be simplified; the word-replacement step uses its output to rewrite the sentence.

It does two things:

- **POS tagging and sentence structure**: labels every word with its part of speech, form, and grammatical role.
- **Complex phrase detection**: finds wordy phrases and suggests simpler ones, such as "to what extent" → "how much".

Built with [spaCy](https://spacy.io) and the [SimplePPDB++](https://github.com/mounicam/lexical_simplification) paraphrase dataset.

## Files

| File | Purpose |
|------|---------|
| `pos_tag.py` | `pos_tag(sentence)`: tags each word |
| `phrase_finder.py` | `find_complex_phrases(sentence)`: finds complex phrases |
| `manual_phrases.csv` | Hand-written phrase list |
| `build_phrase_list.py` | Builds `simpleppdb_phrases.csv` from SimplePPDB++ (run once) |
| `simpleppdb_phrases.csv` | Phrase list from SimplePPDB++ (generated, optional) |
| `test_phrase_finder.py` | Shows detected phrases and the changed sentences |
| `evaluate.py` | Scores the phrase finder against labeled sentences |
| `eval_sentences.csv` | Hand-made labeled sentences (quick checks) |
| `eval_dev.csv` | Labeled real-text sentences for finding problems and tuning (10 paragraphs) |
| `eval_test.csv` | Labeled real-text sentences for the final score only (5 paragraphs) |
| `review_simpleppdb.py` | Lists SimplePPDB phrases found in development text, for review |
| `verified_phrases.csv` | SimplePPDB phrases you reviewed and marked good (generated) |
| `build_asset_data.py` | Builds training data for the phrase detector from ASSET |
| `train_crf.py` | Trains the CRF phrase detector and compares it with baselines |
| `phrase_model.py` | `detect(sentence)`: finds complex phrases with the trained CRF |
| `phrase_replacer.py` | Suggests a replacement for a detected phrase: learned table, then BERT |
| `review_replacements.py` | Collects the tool's replacements for you to judge, then tunes BERT's confidence filter |
| `learned_replacements.csv` | Phrase replacements people made in ASSET's training sentences (generated) |

Keep all files in the same folder.

## Setup

```
pip install spacy
python -m spacy download en_core_web_sm
```

Optional, to add the SimplePPDB++ phrases:

1. Download `simpleppdbpp_xl.tsv.gz` (about 260 MB) from the `SimplePPDBpp` folder at https://github.com/mounicam/lexical_simplification. Use the file's Download button; `git clone` only gets a placeholder unless Git LFS is installed.
2. Run `python build_phrase_list.py simpleppdbpp_xl.tsv.gz` (takes a few minutes).

## Try it

```
python pos_tag.py            # tag sentences interactively
python phrase_finder.py         # find phrases interactively
python test_phrase_finder.py    # before/after on example sentences
```

Example from `test_phrase_finder.py`:

```
1. Before: To what extent is climate change caused by human activity?
   - "To what extent" -> "how much"   [manual]
   After:  How much is climate change caused by human activity?
```

## Usage

```python
from pos_tag import pos_tag
from phrase_finder import find_complex_phrases

tokens = pos_tag("The board made a decision prior to the merger.")
phrases = find_complex_phrases("The board made a decision prior to the merger.")
```

### `pos_tag(sentence)` returns one dict per word

```python
{"index": 2, "text": "made", "lemma": "make", "pos": "VERB", "tag": "VBD",
 "dep": "ROOT", "head": 2, "start": 10, "whitespace": " "}
```

| Field | Meaning |
|-------|---------|
| `index` | Word position (0, 1, 2, ...) |
| `text` / `lemma` | The word, and its base form |
| `pos` | Part of speech: `NOUN`, `VERB`, `ADJ`, `ADV`, `PROPN`, `DET`, ... |
| `tag` | Exact form: `VBD` past verb, `NNS` plural noun, `JJ` adjective, ... |
| `dep` / `head` | Grammatical role (`nsubj`, `dobj`, `ROOT`, ...) and the index of the word it attaches to |
| `start` / `whitespace` | Character position, and the space after the word (for rebuilding text) |

Look up any tag with `spacy.explain("VBD")`.

### `find_complex_phrases(sentence)` returns one dict per phrase

```python
{"text": "made a decision", "suggestion": "decide", "source": "manual",
 "start_index": 2, "end_index": 5, "start_char": 10, "end_char": 25,
 "first_tag": "VBD", "sentence_start": False, "pattern": "make a decision"}
```

| Field | Meaning |
|-------|---------|
| `text` / `suggestion` | The phrase found, and a simpler replacement (base form) |
| `source` | `manual` or `simpleppdb` |
| `start_index` / `end_index` | Word positions, matching `pos_tag` (end is exclusive) |
| `start_char` / `end_char` | Character positions in the sentence |
| `first_tag` | Tag of the first word; use it to fix tense ("made a decision" → "decided") |
| `sentence_start` | `True` if the replacement should be capitalized |

## Phrase lists

- **`manual_phrases.csv`**: about 25 hand-checked phrases. Add more as `complex,simple` lines.
- **Verb patterns** in `phrase_finder.py` (`FLEXIBLE_PATTERNS`): phrases whose verb changes form, like "make / made / makes a decision".
- **"in a ___ way" rule** in `phrase_finder.py`: turns "in a theatrical way" → "theatrically", "in decisive ways" → "decisively", "in a very careful manner" → "very carefully". The adverb is built from the adjective, so one rule covers many phrases. Adjectives with no good adverb ("timely", "new") are skipped; edit `NO_ADVERB` and `ADVERB_EXCEPTIONS` to adjust.
- **`simpleppdb_phrases.csv`**: about 80,000 multi-word phrases filtered from SimplePPDB++. The build script keeps pairs that are clearly simpler (`--min-complexity`, default 1.0) and close in meaning (`--min-ppdb`, default 4.0), and drops contraction fragments, foreign words, one-word swaps, replacements that add a form of "be" ("diplomatic relations" → "diplomacy is"), and padded pairs that share a first or last word ("prior to the" → "before the").

If phrases overlap in a sentence, a manual phrase always wins over a SimplePPDB phrase. Between phrases from the same list, the longer one wins.

## Evaluation

```
python evaluate.py
```

Runs the phrase finder on `eval_sentences.csv` and reports precision (how many detections were correct), recall (how many expected phrases were found), F1, and suggestion accuracy, overall and per source. It also lists every miss, false alarm, and wrong suggestion.

Each row of `eval_sentences.csv` is a sentence plus the phrases that should be detected:

```
"Prior to the exam, we studied.",prior to => before
"Due to the fact that we were unable to go, we stayed.",due to the fact that => because ; were unable to => could not | couldn't
The cat sat on the mat.,
```

Use `=>` before the acceptable replacements, `|` between alternatives, `;` between phrases, and leave the second column empty when nothing should be detected.

### Development and test sets

`eval_dev.csv` and `eval_test.csv` come from 15 paragraphs of real articles, split by article so no source appears in both. Use the development set freely to find problems. Use the test set only for the final comparison, and never add its phrases to the lists; otherwise the score no longer measures how the tool does on new text.

### Reviewing SimplePPDB

```
python review_simpleppdb.py collect eval_dev.csv   # lists SimplePPDB phrases that match
# fill in the "verdict" column of simpleppdb_review.csv with good / bad
python review_simpleppdb.py apply                   # writes verified_phrases.csv
```

`apply` also prints SimplePPDB's precision on the reviewed phrases.

### Comparing setups

```
python evaluate.py eval_test.csv --lists manual
python evaluate.py eval_test.csv --lists manual,verified
python evaluate.py eval_test.csv --lists manual,simpleppdb
```

When phrases overlap, the priority is manual > verified > simpleppdb.

## Trained phrase detector

A CRF model that learns which phrases people simplify, from the ASSET dataset (2,357 sentences, each rewritten by 10 people). A phrase counts as complex when at least 3 of the 10 people replaced it with shorter wording, such as "prior to" → "before" or "provided assistance to" → "helped".

### Setup

```
pip install python-crfsuite wordfreq
git clone https://github.com/facebookresearch/asset.git
python build_asset_data.py asset      # about 30 seconds
python train_crf.py                   # about 2 minutes
```

`build_asset_data.py` writes `asset_phrases.jsonl` (sentences with B/I/O labels, split into train / dev / test) and `learned_replacements.csv`. `train_crf.py` tunes the model on dev, saves `phrase_crf.crfsuite` and `phrase_crf.json`, and reports test scores.

### Use

```python
from phrase_model import detect
detect("The meeting took place prior to the election.")
# [{'start_index': 2, 'end_index': 4, 'text': 'took place'},
#  {'start_index': 4, 'end_index': 6, 'text': 'prior to'}]
```

`detect(sentence, threshold=0.3)` makes it stricter (fewer, more reliable phrases); a lower threshold finds more.

### How it works

Each word gets a label: B (starts a complex phrase), I (inside one), or O. Features for each word include the word and its lemma, suffix, POS tag, dependency role (from `pos_tag.py`'s spaCy model), word frequency, and the neighbouring words. Instead of taking only the single most likely labeling, the model marks words whose probability of being in a complex phrase is above a threshold tuned on the dev split.

### Results (ASSET test split, 359 sentences, 141 labeled phrases)

| Detector | Exact F1 | Overlap P | Overlap R | Overlap F1 |
|---|---|---|---|---|
| Manual list + rules | 0.0% | 50.0% | 2.1% | 4.1% |
| Learned table (lookup of training phrases) | 7.6% | 26.3% | 17.7% | 21.2% |
| **CRF model** | 6.3% | 27.7% | 22.7% | **24.9%** |
| One human vs. the other 9 (ceiling) | 16.7% | 35.0% | 32.7% | 33.7% |

"Exact" requires the same phrase boundaries; "overlap" counts a prediction as correct if it overlaps a labeled phrase. People often disagree about which phrases to simplify, so even a single human only reaches 33.7% overlap F1 against the others; the CRF reaches about three-quarters of that. Your numbers may differ slightly after retraining.

## Replacement stage

When the trained model finds a phrase that the manual list doesn't cover, `phrase_replacer.py` picks a replacement:

1. **Learned table**: if people replaced this exact phrase in ASSET's training sentences, use their most common replacement ("took place" → "happened"). When BERT is installed, the suggestion is used only if BERT agrees: for this exact sentence, BERT must rank it among its own top 10 guesses (`AGREE_TOP`) and it must keep the meaning. "out of" → "from" is right in general but wrong in "out of service", so there BERT should disagree and the phrase falls through to step 2. The review file shows each rejected table suggestion in the `rejected_table` column.
2. **BERT**: otherwise, hide the phrase and let BERT propose 1–2 words that fit the gap. BERT sees the original sentence too, so it stays close to the meaning. Candidates are kept only if they are shorter, use common words, and keep the sentence's meaning (similarity ≥ 0.80 with a sentence-embedding model). The survivors are ranked by BERT's confidence, meaning similarity and word frequency together, following the LSBert method.

3. **Fluency check**: for the chosen replacement (from either step), BERT reads the area around the change, three words on each side, in the original and the rewritten sentence, with nothing hidden. If the rewrite reads much less naturally ("By the mid the 20th century"), it's rejected and the next BERT candidate is tried (up to 5). The allowed drop is tuned from your judgments.

4. **Meaning check**: an NLI (natural language inference) model, `cross-encoder/nli-deberta-v3-small`, reads the original and rewritten sentences in both directions and gives the probability that they mean the same thing. This catches rewrites that are grammatical but wrong ("remained in orbit" → "left orbit"). The minimum is tuned from your judgments, among the replacements that pass the fluency check.

If no candidate passes, the phrase is left alone: it's better to keep a wordy phrase than to replace it with something wrong.

Setup (downloads about 0.5 GB of models the first time):

```
pip install torch transformers sentence-transformers sentencepiece protobuf
python phrase_replacer.py          # demo on a few sentences
```

Without these packages, everything still works; model-detected phrases just use the learned table only.

`find_complex_phrases` now uses the sources `manual`, `verified` and `model` by default (SimplePPDB is off; add it with `PHRASE_LISTS` or `--lists`). For model phrases, `source` is `"model"` and `pattern` is `"model (learned)"` or `"model (bert)"`. The pipeline uses a stricter detector threshold (0.3, `MODEL_THRESHOLD` in `phrase_finder.py`) than training picked (0.1), because 0.1 flagged too many wrong phrases on real articles.

### Judging replacements by hand

The labeled sets only credit phrases someone labeled in advance, but the tool can also make good replacements nobody anticipated ("a single place" → "one place"). So judge its actual output:

```
python review_replacements.py collect eval_dev.csv   # every replacement, before/after
# mark each row of replacements_review.csv as good or bad
python review_replacements.py apply                  # precision per source + tunes the filter
```

`apply` prints the share of good replacements for each source (manual, learned, bert). For two scores, BERT confidence and fluency drop, it picks the cutoff that keeps the most good replacements while rejecting the most bad ones, or none if no cutoff helps. It saves this in `replacer_settings.json`, which `phrase_replacer.py` reads automatically. Each cutoff is placed in the middle of the gap between the last value it keeps and the first it rejects, for some margin.

To see what the tool really outputs with the tuned filters on, and get the precision number for your report:

```
python review_replacements.py collect --final eval_dev.csv   # writes replacements_final.csv
# mark any rows without a verdict
python review_replacements.py apply --final                  # precision only; filters unchanged
```

Compare on your real-text sets:

```
python evaluate.py eval_dev.csv --lists manual
python evaluate.py eval_dev.csv --lists manual,model
```

## Known limitations

- **SimplePPDB++ is noisy.** It was built automatically, so some suggestions change the meaning (e.g. "were unable to" → "could be"). Override bad ones by adding the phrase to `manual_phrases.csv`.
- **No grammar fixing.** Suggestions are base forms; adjusting tense and agreement is the replacement step's job.
- **Coverage.** Phrases not in either list are not detected.

## References

- Qiang, J. et al. (2020). LSBert: A Simple Framework for Lexical Simplification. *arXiv:2006.14939*.
- Alva-Manchego, F. et al. (2020). ASSET: A Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations. *ACL*. (CC BY-NC 4.0)

- Maddela, M. & Xu, W. (2018). A Word-Complexity Lexicon and A Neural Readability Ranking Model for Lexical Simplification. *EMNLP*.
- Pavlick, E. & Callison-Burch, C. (2016). Simple PPDB: A Paraphrase Database for Simplification. *ACL*.
- Honnibal, M. et al. spaCy: Industrial-strength Natural Language Processing in Python.