# Learned structure pilot, October 26

This run compares a learned rewrite ranker, a CRF edit tagger and a locally
fine-tuned T5-small model. The implementation and prediction CLI are opt-in.
The v7 demo keeps its existing behavior and artifacts.

## Results and decision

The full run completed successfully on 128 development and 128 held-out test
sentences. All models, thresholds and the T5 checkpoint were frozen before
test scoring. Test results use the documented single-reference SARI variant.

| System | Raw SARI | Filtered SARI | Filtered edits | Mean longest sentence, words |
|---|---:|---:|---:|---:|
| Unchanged input | 30.22 | 30.22 | 0/128 | 27.5 |
| Existing rules | 30.35 | 30.35 | 2/128 | 27.3 |
| Learned ranker | 52.53 | 45.54 | 70/128 | 20.9 |
| CRF edit tagger | 42.81 | 38.39 | 34/128 | 24.4 |
| T5-small | 57.43 | 43.72 | 52/128 | 22.8 |

The filtered ranker gains 15.20 SARI over the existing rules, with a paired
bootstrap 95% interval of 12.01 to 18.78. T5 gains 13.37, interval 10.26 to 16.88.
The ranker's 1.83-point lead over T5 has interval -1.59 to 5.33, so this sample
does not establish that the ranker outperforms T5. These intervals use 1,000
resamples with seed 4701. The model-to-model comparison is descriptive and was
computed after the frozen run, without changing any model.

T5's raw proposals score highest, but only 55.5% pass the diagnostic checks.
Raw pass rates are 57.0% for the ranker and 81.2% for the editor. T5 raw failures
include 37 argument-role checks, 16 clause checks and 10 content-retention checks;
categories overlap. A failure can reflect a bad rewrite or a conservative parser
check. Even unchanged inputs fail five clause checks, which explains why filtered
pass rates are below 100%.

The ranker is the most practical candidate for the next development iteration,
with a 27 ms median prediction time versus 567 ms for T5 on this shared Mac.
The editor takes 4 ms, but has restricted training coverage and fewer accepted
edits. These timings include candidate generation and checks; T5 uses MPS and
the other models use CPU. They are not controlled hardware comparisons.

Keep all three opt-in. A qualitative spot check of the earliest accepted test
outputs found ranker fragments that passed the clause check. A copied subject
followed by a participle can pass, as can a noun phrase whose only finite verb
belongs to an embedded relative clause. The current check only requires some
subject and some verb in each sentence. This is a known gap, and a higher SARI
score does not resolve it. The frozen scores above describe the measured code;
no post-test guard adjustment is included in those scores.

The next development change should check the main clause's finite predicate,
including copular constructions, on a separate development set. It needs positive
controls for valid splits. Then compare against T5 on an untouched slice and
review meaning and fluency on technical documents before changing the demo.

## Authored diagnostics

The six ungraded examples in `diagnostics.json` show behavior outside the scored
dataset. All three split the sentence about a neural network processing 500
records while retaining the number and technical term. The ranker also copies
the subject when splitting a server's two actions and converts a passive report
sentence to active voice. T5 leaves those two inputs unchanged.

The CRF proposes "She" for the server; the role check rejects it. Its relative
clause example produces a fragment that the checks also reject. The connective
check rejects the editor's "but" to "However" rewrite, showing that a check can
also reject a plausible edit. None of these examples has a human correctness
label, and they do not measure domain performance.

## Training and selection

All models use the same filtered WikiSplit split: 2,000 training pairs, 128
development pairs and 128 test pairs. Training samples come from the first
50,000 released training rows. The sampler removes normalized exact matches
against all eligible development and test sources and full targets before
training. This removed five training-pool rows. The manifest records the pinned
dataset revision, downloaded file hashes and all filter counts.

The ranker learned from 10,293 candidates across the 2,000 training pairs. A
development sweep chose a predicted-gain threshold of 10. Its supervision is
reference overlap, rather than human judgments of correctness.

The CRF's restricted insertion and deletion vocabulary could represent 526 of
the 2,000 pairs, or 26.3%. It trained on those pairs and 131 identity examples
drawn from training reference components. Its insertion vocabulary has 85
entries. Development selection chose confidence threshold zero, with the shared
checks still applied.

T5 uses a separate environment and public pretrained weights. Its final CPU run
uses Adafactor, microbatches of two, two accumulation steps and an effective
batch of four. It completed 400 optimizer updates and saw 1,600 training
examples, less than one full epoch of the 2,000-pair pool.
The checkpoint selector uses development teacher-forced loss.
Test references do not select thresholds or checkpoints.
It selected update 300, with development loss 0.5994; update 400 scored 0.6077.
The selected weights therefore saw 1,200 examples. The original summary counted
one unprocessed batch and reported 1,604 examples for the full run. The published
summary corrects this to 1,600 and preserves the original value with an explanation.
The counter fix does not change training or weights.

The classic models were fitted from commit `a975fa5`. The final T5 run uses
`135010a`. The source provenance record includes full commits, module hashes
and the prepared-data hash. Earlier interrupted attempts are separate from the
final run. An MPS/AdamW attempt reached 100 updates before a resource stop; those
weights were preserved locally and were not used to initialize the final run.

## Resource handling and reproduction

Training runs on the Mac in tmux session `cleartext-structure-20261026`.
Yoga coordinates and stores the copied artifacts. Yoga had about 7 GB free disk,
below its required reserve. The lower-memory CPU configuration has an estimated
3 GB peak RAM requirement. Mac warning pressure pauses training with optimizer
state retained. An external watcher also suspends this job on warning pressure
during generation and resumes it when pressure returns to normal. Other projects'
processes are left running.

The full comparison uses the same saved weights with T5 generation on MPS and
the ranker/editor on CPU. The earlier CPU evaluation stopped at development
sentence 49, before test scoring, because generation was slow. No weights,
thresholds or sample membership changed. MPS startup compiled Metal shaders;
after startup it scored 112 development sentences in about 76 seconds, including
other model predictions. Reported latency is wall time on this shared Mac and
can include memory pauses. It is not an isolated serving benchmark.

The final command, after installing the separate environment described in
[the experiment guide](../../docs/STRUCTURE_EXPERIMENTS.md), is:

```sh
CLEARTEXT_WAIT_FOR_CAPACITY=1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /path/to/structure-venv/bin/python models/scripts/structure_experiment.py all \
  --run /path/to/new-run --cache /path/to/wikisplit-cache \
  --train-size 2000 --dev-size 128 --test-size 128 \
  --max-steps 400 --batch-size 2 --accumulation-steps 2 --device cpu
```

To reproduce the evaluation device, run the stages separately and finish with
`evaluate --run /path/to/run --device mps`. Running `all --device cpu` evaluates
on CPU too. Resource checks and a capacity watcher are needed on a shared host.

The local model-only bundle is `cleartext-structure-models-20261026.tar.gz`,
226,638,290 bytes. SHA256 is
`d886ec69c12b0d2d3142976795c4a79d6ca705f4a765839a6cd0ec954f7f7916`.
Its 11 model/configuration files were verified after packaging. It contains no
training rows or benchmark predictions. The provenance file records individual
artifact hashes; those files matched between the Mac and Yoga copies.

`environment.txt` records the actual installed packages. Source commits and
hashes identify each stage because capacity fixes were added between attempts.
The suite passed 23 focused tests, including an offline T5 optimizer update,
accumulated training, checkpoint reload, alignment round trips, split separation,
modal scope and capacity pause/resume behavior. New Python files passed Ruff.

## Limits

These are noisy, automatically mined general-domain pairs with length and
overlap filters. They do not establish results on technical documents. Exact
match filtering does not remove all paraphrased duplicates or rule out overlap
with pretraining.

SARI here is the documented single-reference, set-based implementation. Bootstrap
intervals describe sampling variation on this filtered test set. They do not
measure correctness. The shared checks depend on spaCy parses, can miss meaning
changes and reject useful paraphrases. Raw and filtered scores must be read
together. The next adoption decision needs human meaning and fluency ratings on
technical sentences, including preserved conditions, quantities and terminology.

Data rows, per-sentence benchmark outputs and weights remain outside Git.
Published examples use the authored diagnostic sentences, which have no gold
ratings and were not used as training references.
