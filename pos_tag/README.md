# Sentence Analysis for Text Simplification

Part of our AI sentence simplification tool. This module analyzes a sentence and finds what can be simplified; the word-replacement step uses its output to rewrite the sentence.

It does two things:

- **POS tagging and sentence structure**: labels every word with its part of speech, form, and grammatical role.
- **Complex phrase detection**: finds wordy phrases and suggests simpler ones, such as "to what extent" → "how much".

Built with [spaCy](https://spacy.io) and the [SimplePPDB++](https://github.com/mounicam/lexical_simplification) paraphrase dataset.

## Files

| File | Purpose |
|------|---------|
| `pos_tagger.py` | `pos_tag(sentence)`: tags each word |
| `phrase_finder.py` | `find_complex_phrases(sentence)`: finds complex phrases |
| `manual_phrases.csv` | Hand-written phrase list |
| `build_phrase_list.py` | Builds `simpleppdb_phrases.csv` from SimplePPDB++ (run once) |
| `simpleppdb_phrases.csv` | Phrase list from SimplePPDB++ (generated, optional) |
| `test_phrase_finder.py` | Shows detected phrases and the changed sentences |

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
python pos_tagger.py            # tag sentences interactively
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
from pos_tagger import pos_tag
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
- **`simpleppdb_phrases.csv`**: about 80,000 multi-word phrases filtered from SimplePPDB++. The build script keeps pairs that are clearly simpler (`--min-complexity`, default 1.0) and close in meaning (`--min-ppdb`, default 4.0), and drops contraction fragments, foreign words, and one-word swaps.

If the same phrase is in both lists, the manual entry wins. If two phrases overlap in a sentence, the longer one wins.

## Known limitations

- **SimplePPDB++ is noisy.** It was built automatically, so some suggestions change the meaning (e.g. "were unable to" → "could be"). Override bad ones by adding the phrase to `manual_phrases.csv`.
- **Padded duplicates.** SimplePPDB++ contains phrases with an extra word attached ("prior to the" → "before the"). Because longer matches win, these can replace a manual match.
- **No grammar fixing.** Suggestions are base forms; adjusting tense and agreement is the replacement step's job.
- **Coverage.** Phrases not in either list are not detected.

## References

- Maddela, M. & Xu, W. (2018). A Word-Complexity Lexicon and A Neural Readability Ranking Model for Lexical Simplification. *EMNLP*.
- Pavlick, E. & Callison-Burch, C. (2016). Simple PPDB: A Paraphrase Database for Simplification. *ACL*.
- Honnibal, M. et al. spaCy: Industrial-strength Natural Language Processing in Python.