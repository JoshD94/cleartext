#!/usr/bin/env python3
"""Train the connector split tagger on WikiSplit train and report tune scores.

Connector-level scores on tune only count sentences with a clean labelled
split (see `wikisplit_label`); other sentences have no reliable negatives.
The saved model is a local artifact and stays outside Git.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from cleartext.features import nlp
from cleartext.structure.boundary_tagger import CONNECTORS, BoundaryTagger, features, wikisplit_rows


def examples(rows, unlabelled_weight):
    X, y, w, groups = [], [], [], []
    docs = nlp().pipe((r["source"] for r in rows), batch_size=256)
    for n, (row, doc) in enumerate(zip(rows, docs)):
        joins = [t for t in doc if t.lower_ in CONNECTORS]
        gold = row["connector_occurrence"]
        if gold is not None and gold >= len(joins):
            continue
        for k, join in enumerate(joins):
            if join.i == 0:
                continue
            X.append(features(doc, join))
            y.append(int(k == gold))
            w.append(1.0 if gold is not None else unlabelled_weight)
            groups.append((n, gold is not None))
    return X, np.array(y), np.array(w), groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--tune", required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--unlabelled-weight", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    train_rows = wikisplit_rows(args.train, args.limit, args.stride)
    X, y, w, _ = examples(train_rows, args.unlabelled_weight)
    vectorizer = DictVectorizer()
    model = LogisticRegression(max_iter=3000, C=0.5)
    model.fit(vectorizer.fit_transform(X), y, sample_weight=w)
    tune_rows = wikisplit_rows(args.tune)
    tX, ty, _, tgroups = examples(tune_rows, args.unlabelled_weight)
    p = model.predict_proba(vectorizer.transform(tX))[:, 1]
    labelled = np.array([g[1] for g in tgroups])
    curve = {}
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        hit = p >= threshold
        tp = int((hit & (ty == 1)).sum())
        curve[str(threshold)] = {
            "labelled_precision": tp / max(1, int((hit & labelled).sum())),
            "labelled_recall": tp / max(1, int(ty.sum())),
            "unlabelled_flag_rate": float(hit[~labelled].mean()) if (~labelled).any() else 0.0,
        }
    weights = sorted(zip(model.coef_[0], vectorizer.feature_names_), reverse=True)
    report = {
        "train_sentences": len(train_rows),
        "train_connectors": len(y), "train_positives": int(y.sum()),
        "tune_sentences": len(tune_rows), "tune_connectors": len(ty), "tune_positives": int(ty.sum()),
        "stride": args.stride, "unlabelled_weight": args.unlabelled_weight,
        "threshold": args.threshold, "curve": curve,
        "top_split": [[n, round(c, 3)] for c, n in weights[:15]],
        "top_keep": [[n, round(c, 3)] for c, n in weights[-15:]],
        "seconds": round(time.monotonic() - start, 1),
    }
    BoundaryTagger(vectorizer, model, args.threshold).save(args.out, report=report)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
