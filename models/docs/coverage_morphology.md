# Repairing incompatible source verb lemmas

The cached generator analyzes `taping` as `tap` and `scraping` as `scrap`. The installed inflection dictionary gives `tapping` and `scrapping` for those lemmas. It supports `tape -> taping` and `scrape -> scraping` instead. The original analyses therefore generate candidates from the wrong verb inventories.

`candidate_morphology.py` adds an explicit wrapper around an existing WordNet generator. It repairs an analysis only when the original lemma has a known dictionary form that disagrees with the observed word, and exactly one alternative dictionary lemma has both an exact WordNet verb entry and the required surface form. The repair keeps POS, tense and the fallback setting. Unknown morphology and ambiguous alternatives keep the original analysis. Equivalent sense inventories also keep the original, as with `enrol` and `enroll`.

Changed analyses regenerate candidates with the corrected lemma's actual WordNet edges. Old candidates and origin senses do not carry over. Unchanged analyses return the original candidate list and order. The wrapper preserves the member generator's source restrictions, phrase support and inflection function. Repaired cases use sorted dictionary lemma order so WordNet pointer enumeration cannot change tied ranks across processes.

The wrapper has no default loader registration. An explicit caller can use it with an existing pipeline:

```python
from cleartext.candidate_morphology import RoundTripWordNetGenerator

pipe.generator = RoundTripWordNetGenerator(pipe.generator)
```

The existing sense scorer receives the corrected target tuple. The meaning floor and acceptance thresholds still apply. Some existing static-vector members read the tagger's lemma directly, so repairing the generator does not correct every feature. Dictionary support does not establish retained sentence meaning.

## Local resource audit

The model-free audit reads the already installed LemmInflect resources and WordNet ZIP directly. It uses no parser, encoder, fitted classifier or downloaded data. Repair decisions finish before it joins existing BenchLS list membership.

All 929 BenchLS cases are reused development. The audit changes two analyses, leaves 225 valid inflected-verb round trips alone, retains one equivalent spelling analysis, and leaves the other 701 cases outside this bounded repair. The corrected dictionary inventories contain `recording` for `taping`, and `collecting`, `grating`, `rubbing` and `scratching` for `scraping`. Both cases previously lacked a generated listed substitute. These are dictionary-supported availability findings. The separate native comparison below measures accepted selections.

The audit used 46,776 kB peak memory according to `/proc/self/status`. Twenty-four pure mechanism tests passed. They cover valid irregular forms, missing dictionary evidence, ambiguous alternatives, exact sense support, equivalent spelling inventories, candidate provenance, inflection and phrase restrictions, deterministic ordering, unchanged candidate identity, leakage checks and the fixed adoption bar. None supplies semantic training labels.

Raw records, fingerprints, protocol and handoff are outside Git in `/home/joshuadirga/.local/state/cleartext/improvement-sprint-20261028/candidate-coverage`. The completed audit is `audit-final-r2`. Earlier attempts remain separate, including an incomplete `audit-final` caused by a relative source-snapshot path in a `runpy` invocation. The script now resolves that path.

## Fixed native comparison

`candidate_coverage_compare.py` first verifies audit inputs, source hashes, the frozen 52-column baseline table and the corrected entity-guard scan. Its metadata-only `--preflight` path fingerprints 33 input files, 73 source files and 107 inference resource files in this checkout. The real preflight and a second hash validation passed with no ML imports, one thread and 23,056 kB peak resident memory under a 300 MiB address-space cap.

The input freeze explicitly includes `runs/initial-20260924/word_model.pkl`, `runs/initial-20260924/context_model.pkl` and `runs/context-20260924/phrase_model.pkl`. The phrase model loads with the pipeline even though this experiment does not call its predictor. The stacker, sense-model and checker defaults also enter the freeze, along with artifact paths named in the run configuration and the original audit's dictionary inputs.

The resource freeze records the local BERT encoder weights, tokenizer files and configuration, both installed spaCy model packages, LemmInflect dictionaries, overrides and both OOV models, English wordfreq tables, the existing Brown bigram cache, WordNet, CMU pronunciations and stopwords. It checks each resource directory's file inventory too. Cache symlinks retain their lookup paths so changing their target bytes invalidates the freeze. Generated `__pycache__`, `.pyc` and `.pyo` files are excluded. Brown, SemCor and OMW corpora are not inference inputs here. The Brown cache must already exist, SemCor training evidence is in the saved sense model, and this run uses English WordNet entries.

Native encoder loading uses the recorded local snapshot rather than resolving the mutable Hugging Face `main` reference again. NLTK lookup is restricted to the recorded directory, which defaults to `~/nltk_data`; `--nltk-data` selects another directory. Package locations come from the invoking interpreter's import metadata. The freeze records Python and the main numerical/NLP dependency versions without importing those packages. It does not hash every installed dependency binary or promise numerical equality across different machines.

After native scoring returns, the script rechecks all frozen input, source and resource hashes, resource and source file inventories, NLTK resource selection, runtime versions and the saved run manifest. A mismatch raises an error before `summary.json` can report success. A completed summary records `freeze_validated_after_native: true`.

These new hashes identify the components used by the new comparison. The historical baseline did not record these implicit model/resource hashes, so they cannot establish an exact byte match to the models that produced its old cached features. The three unchanged native controls remain required spot checks. Reconstructing probabilities from saved baseline features does not prove historical model-byte equality. Both the new manifest and any completed summary record `historical_model_bytes_verified: false`.

Both policies use the same corrected entity guard. Another sprint worker added code-literal protection to the current checkout, so the comparison uses an explicitly supplied snapshot from commit `fde61c46253687abe03d9d6207023090bab00937`. Its SHA-256 matches the completed corrected-guard scan. The snapshot affects only this experiment. It does not replace the current checkout's guard.

The native run checks raw/native repair decisions on every case, exact verb inventories and direct relations on repaired cases, three fixed unchanged cache rows, and batch/single scores on each repaired case. It scores every repaired candidate with the existing frozen components. Existing list labels enter after feature construction.

The comparison reconstructs the original baseline models from their saved folds and original candidate rows only. It must reproduce every saved baseline probability within `1e-12`. Repaired candidates never enter training. Their target groups and normalized complete contexts must both be absent from each training fold. The historical baseline retains its original target-word folds; this experiment makes no broader claim of context separation for those baseline rows.

Selection uses the unchanged `0.10` meaning floor and thresholds `0.45` and `0.80` across all three repeats. Every unchanged case must retain its selected word exactly. Adoption requires mean default net gain of at least five, plus at least 90% listed-match precision and 20% edit coverage in every strict repeat. Log loss across different candidate populations cannot qualify the change. With only two affected cases, the maximum paired net gain is four. This narrow repair therefore cannot pass the fixed adoption bar, even if both cases improve. There is no threshold search, new saved run or automatic promotion.

Run from the repository root. Use a new output directory for each execution:

```sh
python3 models/scripts/experiments/candidate_coverage_roundtrip.py \
  --output /path/to/new-audit

models/.venv/bin/python models/scripts/experiments/candidate_coverage_compare.py \
  --audit /path/to/new-audit \
  --protocol /home/joshuadirga/.local/state/cleartext/improvement-sprint-20261028/candidate-coverage/protocol.json \
  --guards /home/joshuadirga/.local/state/cleartext/eligibility-audit-20261028/guard-comparison-yoga \
  --guard-source /home/joshuadirga/.local/state/cleartext/improvement-sprint-20261028/candidate-coverage/guard-source.py \
  --output /path/to/new-native-comparison
```

Append `--preflight` for the lightweight check. To save a freeze without native loading, also pass `--write-freeze-manifest /path/to/new-freeze.json`. The destination must be a new file. For a pinned reproduction, pass `--freeze-manifest /path/to/new-freeze.json` to either the preflight or native command. The script requires identical input, source, resource and runtime records, and uses the prior encoder snapshot even if the cached `main` reference moves. Older incomplete manifests are rejected. `--encoder-snapshot` can explicitly select a local BERT snapshot when preparing a new freeze. A changed source tree or runtime requires a new reviewed freeze.

The checked freeze is `/home/joshuadirga/.local/state/cleartext/improvement-sprint-20261028/candidate-coverage/native-freeze-manifest-fix.json`. The state launcher `run-native.sh` requires it. Seven standard-library manifest tests in `models/tests/test_candidate_coverage_manifest.py` require both implicit scoring models and the phrase model, reject changed bytes and missing files, check resource inventory and cache symlink changes, check pinned reproduction, and require hash validation before a successful native summary. The native call in these tests is a stub.

## Completed native result

The parent ran the comparison on the Mac through Yoga tmux `cleartext-candidate-roundtrip-20261028`. Yoga had only 4.8 GB available and could not fit the estimated 3 GB job with the required reserve. The Mac had normal memory pressure, 71% free memory and 6.7 GiB disk. All models were already staged, so the native run needed under 100 MB additional disk. A parent monitor checked pressure and disk every five seconds and could stop only this job's process group. It recorded normal pressure throughout. The job exited successfully in 42.87 seconds, with 2,266,759,168 bytes maximum resident memory. The scoring stage itself took 40.62 seconds.

Before launch, 185 copied files and all 73 sources matched the Yoga freeze. A fresh model-free audit reproduced all 929 records and the complete JSONL hash exactly. The installed parser, LemmInflect and wordfreq resources also matched Yoga bytes. The new Mac freeze records its runtime differences, including NumPy 2.5.4, NLTK 3.10.4 and Torch 2.14.1. Its SHA-256 is `f811b88007c3bf767eaff6b95260d32a8650babba76c812db5e47dcae901d2a8`. The original Yoga freeze remains unchanged.

Native repair decisions matched the raw audit on all 929 cases. All three fixed unchanged controls and both repaired-case batch/single checks passed their original tolerances. The run scored 27 regenerated candidates. All three baseline reconstructions passed the fixed `1e-12` probability tolerance; the largest absolute difference was `1.99e-13`. Post-run artifact validation passed. The pulled result files and all 73 saved source files were hash-verified on Yoga.

| Seed | Default listed edits, before to after | Default unlisted edits | Strict listed edits, before to after | Strict unlisted edits |
| --- | --- | --- | --- | --- |
| 0 | 515 to 517 | 128 unchanged | 279 to 280 | 28 unchanged |
| 1 | 511 to 513 | 128 unchanged | 266 to 267 | 27 unchanged |
| 2 | 516 to 518 | 131 unchanged | 271 to 272 | 29 unchanged |

At the default threshold, both repaired cases change from abstention to a listed substitute in every repeat: `taping` to `recording`, and `scraping` to `scratching`. Strict selection adds only `recording`. All 927 unchanged cases keep their original selections. Strict listed-match precision is 90.37% to 90.91%, at 31.65% to 33.15% edit coverage. These lexical list-match figures do not measure full-sentence semantic accuracy.

The mean default net gain is two. It fails the fixed requirement of five even though the strict precision and coverage gates pass. The repair stays opt-in; no model variant was saved or promoted. Aggregate results, the actual Mac freeze and staging verification are in [candidate evidence](../evidence/candidate-roundtrip-20261028/README.md). Full raw outputs and logs are in the state directory under `native-mac-v1`.

No SWORDS labels, TSAR data, reserved domain test, new human reviews, synthetic semantic labels or paid services enter this work. The resource audit and mechanism tests do not measure full-sentence accuracy.
