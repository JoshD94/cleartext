# One research paper through ClearText, November 16, 2026

How does the current winner behave on a whole real paper, sentence by
sentence, and how does it compare with the two variants the November 10 study
rejected? This folder holds the scripts' outputs that are safe for Git:
aggregates only. The paper text, every output, the reading page and the rating
form stay local under `~/.local/state/cleartext/paper-run-20261116/`. No LLM
touches the text anywhere.

**Paper.** Alva-Manchego, Martin, Bordes, Scarton, Sagot and Specia, "ASSET: A
Dataset for Tuning and Evaluation of Sentence Simplification Models with
Multiple Rewriting Transformations", ACL 2020,
[2020.acl-main.424](https://aclanthology.org/2020.acl-main.424/). Licence
checked on the Anthology page: material from 2016 on is CC BY 4.0. PDF sha256
`0084e6dac4785c1db2046cb2b54f41ae484912ee9733a42e1ea35014e1967329`.

## Result

**On a real paper the winner barely acts, and what it does is safe by every
automatic check.** It changes 8 of 227 sentences (3.5%; 8 of 76 paragraphs).
Whole-paper FK goes from 13.28 to 13.10, and the changed sentences go from
13.95 to 9.98. Their NLI is 0.982 on average (lowest 0.960). No sentence is
flagged, and no number, negation or entity is lost. All 8 edits are splits:
5 by the compound splitter, 3 by T5. The authored phrase rules never fired.

| 227 sentences, 76 paragraphs | Winner | T5 unchecked | v7 words (17 paragraphs only) |
|---|---:|---:|---:|
| Sentences changed | 8 (3.5%) | 83 (36.6%) | 16 of 49 (32.7%) |
| FK, scored text (original → output) | 13.28 → 13.10 | 13.28 → 12.10 | 14.42 → 14.27 |
| FK, changed sentences | 13.95 → 9.98 | 13.86 → 10.91 | 16.83 → 16.02 |
| Sentence NLI, changed (mean / min) | 0.982 / 0.960 | 0.929 / 0.479 | 0.856 / 0.192 |
| Paragraph NLI (mean / min) | 0.983 / 0.924 | 0.964 / 0.507 | 0.904 / 0.147 |
| Flags: NLI < 0.80 / number lost / negation lost | 0 / 0 / 0 | 7 / 1 / 0 | 4 / 0 / 0 |
| Kea-Roy S / U, changed sentences | 0.301 / 0.286 | 0.295 / 0.271 | 0.268 / 0.238 |

On the same 17 paragraphs, the winner changes nothing, T5 unchecked changes 20
of 49 sentences (2 flagged), and v7 words changes 16 (4 flagged; paragraph NLI
0.904 against the winner's 0.983). The full tables are in
[paper-summary.md](paper-summary.md) and
[paper-summary.json](paper-summary.json), which include the per-section FK.
Passage NLI abstains on 2 paragraphs of 187 and 207 words that exceed
DeBERTa's 512 tokens; sentence NLI never abstains.

What the reading page shows (local):

- **The checks are why the winner is safe, and why it does little.** T5
  without checks lowers whole-paper FK by 1.2 grades, but 7 changed sentences
  fall under NLI 0.80. In those sentences it silently deletes characters
  outside its vocabulary (κ, §, ±, accents in "Aluísio"), cuts citations, and
  once drops a number. This matches the November 10 ablation (−1.2 grade,
  fails numbers).
- **NLI misses fluency breaks.** One winner compound split reads "We randomly
  chose 100 original sentences from ASSET. , for each of them, we sampled one
  system simplification." It scores NLI 0.99 and no flag. Human fluency
  ratings are the check that would catch this.
- **v7 words costs meaning on technical prose** and barely moves FK (−0.15 on
  the text it scored).
- The pipeline's spaCy sentencizer sometimes splits inside citations ("…;
  Aluísio et al., 2008; …)") and splits the title at its colon. Sentence
  counts and per-sentence scores follow those splits.

**v7 words was stopped after 18 paragraphs.** On a loaded Yoga a v7 paragraph
took 77 to 1,860 s, because BERT ranks every flagged word. The other 58
paragraphs projected to about 30 hours. v7 already fails the November 10
gates, so it was not allowed to block this run. `batch.sh` needs Yoga-only
ignored artifacts (the T5 checkpoint, the v7 run, `models/data`), so it cannot
use `--sync` on the Mac, and the remaining paragraphs were dropped. Its paper
numbers cover p000–p016 (17 paragraphs with paragraph metrics, 49 sentences).
The rating kit takes most v7 pairs from the panel.

## Method

1. **Extraction** ([`evals/paper_extract.py`](../../paper_extract.py)):
   `pdftohtml -xml` (poppler) plus fixed font/position rules. It keeps the
   title, abstract and every section through Acknowledgements. It drops
   references, tables, figure text, captions, display equations, footnotes and
   page numbers. Line-end hyphens are joined when the document or wordfreq
   knows the joined word, and kept between two real words (`multi-reference`).
   Run-in bold labels ("Qualification Test.") are kept as labels, not
   simplified. 76 paragraphs, 4,756 words.
2. **Generation**: the paragraphs are a panel of the ensemble study
   (`ensemble_study freeze --panel paper=...`, new in this PR). Each config runs
   on whole paragraphs, so each stage sees its paragraph context and the
   per-paragraph edit budgets match the study. Outputs are checkpointed per
   paragraph and T5 is memoized per sentence. Same frozen artifacts as
   November 10 (T5 step-441, WikiSplit boundary tagger, v7 run, DeBERTa NLI).
3. **Sentences** ([`evals/paper_run.py`](../../paper_run.py) `score`): each
   paragraph is split with the pipeline's own spaCy sentencizer. Every stage
   snapshot (after T5, after the compound splitter, after phrases, after words)
   is aligned back to those sentences (difflib word alignment, snapped to the
   output's sentence starts), so each sentence gets its output and the stages
   that changed it.
4. **Metrics per sentence**: FK grade; DeBERTa NLI both ways (the study's
   model and memo; mean of the two directions); numbers, negations and
   entities lost (the `preservation.py` extractors); Kea-Roy's evaluator
   (`evaluate_base.evaluate_record` with `trained_weights.json`: S, U,
   P_meaning). Per paragraph: the study's full scorer (passage NLI both ways,
   explicit checks). Per section and whole paper: FK of the joined text.
5. **Flags**: a changed sentence with NLI below 0.80, or a lost number or
   negation.

Paragraph metrics come from `ensemble_study run --panel paper`. Sentence rows
are bound to the study's signatures and to the hashes of `paper_run.py`,
`evaluate_base.py` and `trained_weights.json`. Queue: personal `later` priority
on Yoga with one thread; IDs and receipts are in provenance.json.

Configs: **winner** = `words=off, phrases=authored, structure=t5+compound_tagger`;
**t5_unchecked** = the same with `structure=t5_unchecked` (T5 with no checks);
**v7_words** = the winner with `words=v7`. Both comparisons are the November 10
single-factor ablations from the winner.

## Human rating kit

[`evals/rating_kit.py`](../../rating_kit.py) samples 50 blind pairs (seed
20261116). 20 are paper sentences: winner 7, T5 unchecked 6, v7 words 3 and
unchanged controls 4. The other 30 are panel passages from the cached
November 10 outputs: 10 / 8 / 8 / 4. Each condition is an independent random
draw from that config's own changed outputs, stratified by paper section or
panel domain, so no condition is limited to what another left over. An
original can therefore appear under two conditions (3 repeats here), never
twice in one. The order is shuffled. There are about 4,250 words to read,
which is roughly an hour per rater. Raters score
simplicity, meaning kept and fluency/grammar from 1 to 5, with an optional
comment. The form is one local HTML file: it autosaves in the browser and
exports `cleartext-ratings-<name>.csv`. The key with conditions and metrics
stays local.

[`evals/rating_score.py`](../../rating_score.py) takes the key and the three
CSVs. It reports Krippendorff's alpha per dimension (interval and ordinal),
human means per condition, and the Pearson and Spearman correlation of each
automatic metric (FK change, FK after, Kea-Roy S and U, NLI) with the mean
human rating. Tested against Krippendorff's (2011) published example
(nominal 0.743, interval 0.849) and on fake rater CSVs.

## Reproduce

```bash
# local: extraction (no model)
pdftohtml -xml -i -q -fontfullname asset.pdf $ST/source/paper
python -m evals.paper_extract $ST/source/paper.xml $ST/paragraphs.json
# queue: one bounded batch at a time until the sentence receipt says complete
job submit --kind script --name cleartext-paper-run --ram 4.5 --minutes 30 --needs yoga \
  --priority later --restartable --owner <thread> --cwd "$PWD" --cmd "$PWD/evals/reports/20261116-paper-run/batch.sh"
# local: aggregates, reading page, rating kit
python -m evals.paper_run report --plan $ST/run1/plan.json --state $ST/run1 --out evals/reports/20261116-paper-run
python -m evals.paper_page $ST/run1/paper-rows.json evals/reports/20261116-paper-run/paper-summary.json $ST/paper-reading.html
python -m evals.rating_kit --paper-rows $ST/run1/paper-rows.json --panel-state ~/.local/state/cleartext/ensemble-v2-20261110/run1 --out $ST/rating-kit
python -m evals.rating_score --key $ST/rating-kit/rating-key.json joshua.csv kea-roy.csv chhoeng.csv
```

`ST=~/.local/state/cleartext/paper-run-20261116`; Python is
`models/.venv/bin/python` with `PYTHONPATH=$CLEARTEXT_EVAL_DEPS:models/src:.`
(see `batch.sh`). Provenance: [provenance.json](provenance.json).

This is one inspected paper, not a test set. FK, NLI and Kea-Roy's scores are
automatic signals, not percentages of correct simplifications.
