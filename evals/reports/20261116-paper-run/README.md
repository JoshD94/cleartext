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

RESULTS_PLACEHOLDER

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

Configs: **winner** = `words=off, phrases=authored, structure=t5+compound_tagger`;
**t5_unchecked** = the same with `structure=t5_unchecked` (T5 with no checks);
**v7_words** = the winner with `words=v7`. Both comparisons are the November 10
single-factor ablations from the winner.

## Human rating kit

[`evals/rating_kit.py`](../../rating_kit.py) samples 50 blind pairs (seed
20261116): 20 paper sentences (winner 7, T5 unchecked 5, v7 words 5,
unchanged controls 3) and 30 panel passages from the cached November 10
outputs (10 / 8 / 8 / 4). Each original appears once, strata are paper
sections and panel domains, and the order is shuffled. Raters score
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
# queue: bounded batches until sentence scoring is complete
evals/reports/20261116-paper-run/drive.sh <owner-thread-id> 4.5
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
