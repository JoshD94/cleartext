"""
Train the CRF complex-phrase detector and compare it with simpler baselines.

Needs asset_phrases.jsonl and learned_replacements.csv from build_asset_data.py.

Usage:
    python train_crf.py

What it does:
  1. Trains a CRF on the train split, trying a few settings (regularization and
     confidence threshold) and keeping the one that scores best on the dev split.
  2. Saves the model as phrase_crf.crfsuite and the chosen threshold as
     phrase_crf.json (both used by phrase_model.py).
  3. Scores three detectors on the held-out test split:
       manual list    - the hand-written manual_phrases.csv + rules
       learned table  - exact lookup of phrases seen in training
       CRF model      - the trained model
  4. Prints the features the model relies on most.

Scores (phrase level):
  exact   - predicted phrase has exactly the same words as a labeled phrase
  overlap - predicted phrase overlaps a labeled phrase (boundaries may differ)
"""

import csv
import json
import os
from collections import Counter

import pycrfsuite
from spacy.matcher import PhraseMatcher

from phrase_model import (MODEL_FILE, SETTINGS_FILE, analyze, decode,
                          labels_to_spans, sentence_features)
from pos_tag import nlp

C1_VALUES = [0.05, 0.2, 0.5]          # L1 regularization: higher = fewer features used
C2_VALUES = [0.01, 0.05, 0.1]         # L2 regularization: higher = smaller weights
THRESHOLDS = [0.5, 0.3, 0.2, 0.15, 0.1]   # confidence needed to mark a word


def load(path="asset_phrases.jsonl"):
    data = {"train": [], "dev": [], "test": []}
    print("Analyzing sentences with spaCy...")
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc = analyze(row["tokens"])
            data[row["split"]].append((doc, sentence_features(doc), row["labels"]))
    for split, rows in data.items():
        print(f"  {split:<5} {len(rows):,} sentences")
    return data


def score(gold_spans, pred_spans):
    """Phrase-level precision / recall / F1, exact and overlap."""
    tp = sum(len(set(g) & set(p)) for g, p in zip(gold_spans, pred_spans))
    n_pred = sum(len(p) for p in pred_spans)
    n_gold = sum(len(g) for g in gold_spans)

    def overlaps(span, others):
        return any(span[0] < o[1] and o[0] < span[1] for o in others)
    tp_pred = sum(overlaps(s, g) for g, p in zip(gold_spans, pred_spans) for s in p)
    tp_gold = sum(overlaps(s, p) for g, p in zip(gold_spans, pred_spans) for s in g)

    def prf(tp_p, tp_g):
        p = tp_p / n_pred if n_pred else 0.0
        r = tp_g / n_gold if n_gold else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f
    return {"exact": prf(tp, tp), "overlap": prf(tp_pred, tp_gold), "n_pred": n_pred, "n_gold": n_gold}


def train(rows, c1, c2, filename):
    trainer = pycrfsuite.Trainer(verbose=False)
    for _, feats, labels in rows:
        trainer.append(feats, labels)
    trainer.set_params({"c1": c1, "c2": c2, "max_iterations": 150,
                        "feature.possible_transitions": True})
    trainer.train(filename)
    tagger = pycrfsuite.Tagger()
    tagger.open(filename)
    return tagger


def crf_predict(tagger, rows, threshold):
    return [decode(tagger, feats, threshold) for _, feats, _ in rows]


def gold(rows):
    return [labels_to_spans(labels) for _, _, labels in rows]


def table_predict(rows, phrases):
    """Baseline: mark any phrase that appeared in the training labels."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("P", list(nlp.tokenizer.pipe(phrases)))
    out = []
    for doc, _, _ in rows:
        spans = sorted(((s, e) for _, s, e in matcher(doc)), key=lambda x: (x[0], -x[1]))
        chosen, last = [], -1
        for s, e in spans:                         # keep longest, non-overlapping
            if s >= last:
                chosen.append((s, e))
                last = e
        out.append(chosen)
    return out


def manual_predict(rows):
    """Baseline: the hand-written manual list + rules from phrase_finder.py."""
    os.environ["PHRASE_LISTS"] = "manual"
    from phrase_finder import find_complex_phrases
    return [[(p["start_index"], p["end_index"]) for p in find_complex_phrases(doc.text)]
            for doc, _, _ in rows]


def show(name, s):
    (ep, er, ef), (op, orr, of) = s["exact"], s["overlap"]
    print(f"  {name:<22} exact P {ep:6.1%} R {er:6.1%} F1 {ef:6.1%}   |   "
          f"overlap P {op:6.1%} R {orr:6.1%} F1 {of:6.1%}   ({s['n_pred']} predicted)")


def main():
    data = load()
    dev_gold = gold(data["dev"])

    print("\nTuning on the dev split:")
    best = None
    for c1 in C1_VALUES:
        for c2 in C2_VALUES:
            tagger = train(data["train"], c1, c2, "tuning.crfsuite")
            for t in THRESHOLDS:
                s = score(dev_gold, crf_predict(tagger, data["dev"], t))
                show(f"c1={c1} c2={c2} t={t}", s)
                if best is None or s["overlap"][2] > best[0]:
                    best = (s["overlap"][2], c1, c2, t)
    os.remove("tuning.crfsuite")
    _, c1, c2, threshold = best
    print(f"\nBest setting: c1={c1}, c2={c2}, threshold={threshold}. Saving {MODEL_FILE.name}")
    tagger = train(data["train"], c1, c2, str(MODEL_FILE))
    SETTINGS_FILE.write_text(json.dumps({"c1": c1, "c2": c2, "threshold": threshold}))

    with open("learned_replacements.csv", newline="", encoding="utf-8") as f:
        table = [r["complex"] for r in csv.DictReader(f)]

    test_gold = gold(data["test"])
    print(f"\nTest split ({len(data['test'])} sentences, {sum(map(len, test_gold))} labeled phrases):")
    show("manual list", score(test_gold, manual_predict(data["test"])))
    show("learned table", score(test_gold, table_predict(data["test"], table)))
    show("CRF model", score(test_gold, crf_predict(tagger, data["test"], threshold)))

    info = tagger.info()
    print("\nFeatures that most strongly signal the START of a complex phrase:")
    top = Counter({feat: w for (feat, label), w in info.state_features.items() if label == "B"})
    for feat, w in top.most_common(15):
        print(f"  {w:6.2f}  {feat}")

    print("\nExamples from the test split (gold = labeled, pred = CRF):")
    shown = 0
    for (doc, feats, labels), g in zip(data["test"], test_gold):
        p = decode(tagger, feats, threshold)
        if (g or p) and shown < 8:
            print(f"  {doc.text[:110]}")
            print(f"     gold: {[doc[a:b].text for a, b in g]}   pred: {[doc[a:b].text for a, b in p]}")
            shown += 1


if __name__ == "__main__":
    main()
