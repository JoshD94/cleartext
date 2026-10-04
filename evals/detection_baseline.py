#!/usr/bin/env python3
"""Detection baseline: CWI-2018 complex-word detection with trained A-E weights.

Methodology:
  1. Load CWI-2018 English train/dev via datasets.py.
  2. Score each target word with C(w,s) using the trained word weights from
     trained_weights.json (evaluate_base.word_complexity_in_context).
  3. Tune the decision threshold on TRAIN (max F1), report train + held-out DEV.
  4. Write results/detection_baseline.json (read live by build_status.py).

Also reports benchmark loader row counts for TSAR-2022 / MultiLS (ranking
evaluation needs a candidate generator from the models side -- pending).

Usage:
    ./venv/bin/python evals/detection_baseline.py [--train-n N] [--dev-n N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import (  # noqa: E402
    load_cwi2018,
    load_multils_en_lcp,
    load_multils_en_ls,
    load_tsar2022_en,
)
from detection import detection_metrics  # noqa: E402
from evaluate_base import word_complexity_in_context, words  # noqa: E402

BENCH = HERE / "data" / "benchmarks"


def make_scorer(word_model: str, word_w):
    if word_model == "gbm":
        from word_models import word_complexity_gbm  # noqa: E402

        def scorer(target, sent_words):
            return word_complexity_gbm(target, sent_words)

        return scorer, "gbm (word_gbm.joblib, 6 features)"

    def scorer(target, sent_words):
        return word_complexity_in_context(target, sent_words, word_w)

    return scorer, "linear (trained A-E weights)"


def score_rows(rows, scorer):
    scored = []
    for r in rows:
        ws = words(r.sentence)
        c = scorer(r.target, ws)
        scored.append((c, r.binary))
    return scored


def f1_at(scored, thr):
    y_true = [b for _, b in scored]
    y_pred = [1 if c > thr else 0 for c, _ in scored]
    m = detection_metrics(y_true, y_pred)["class_1"]
    return m["precision"], m["recall"], m["f1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=None)
    ap.add_argument("--dev-n", type=int, default=None)
    ap.add_argument("--word-model", choices=("linear", "gbm"), default="linear",
                    help="word scorer: trained A-E linear weights or the GBM model")
    args = ap.parse_args()

    word_w = json.loads((HERE / "trained_weights.json").read_text())["word_weights"]
    scorer, scorer_desc = make_scorer(args.word_model, word_w)
    print(f"word model: {scorer_desc}")

    train_rows = load_cwi2018(BENCH / "cwi2018_en_train.tsv")
    dev_rows = load_cwi2018(BENCH / "cwi2018_en_dev.tsv")
    if args.train_n:
        train_rows = train_rows[: args.train_n]
    if args.dev_n:
        dev_rows = dev_rows[: args.dev_n]

    print(f"scoring {len(train_rows)} train + {len(dev_rows)} dev CWI instances...")
    train_scored = score_rows(train_rows, scorer)
    dev_scored = score_rows(dev_rows, scorer)

    best_thr, best_f1 = 0.5, -1.0
    thr = 0.10
    while thr <= 0.90:
        _, _, f1 = f1_at(train_scored, round(thr, 2))
        if f1 > best_f1:
            best_f1, best_thr = f1, round(thr, 2)
        thr += 0.05

    out = {
        "word_model": args.word_model,
        "word_model_desc": scorer_desc,
        "threshold_tuned_on": "cwi2018_en_train",
        "threshold": best_thr,
        "train": dict(zip(("precision", "recall", "f1"), f1_at(train_scored, best_thr))),
        "dev": dict(zip(("precision", "recall", "f1"), f1_at(dev_scored, best_thr))),
        "train_n": len(train_rows),
        "dev_n": len(dev_rows),
        "ranking_benchmarks": {
            "tsar2022_en_test_gold": len(load_tsar2022_en(BENCH / "tsar2022_en_test_gold.tsv")),
            "multils2024_en_trial_ls": len(load_multils_en_ls(BENCH / "multils2024_en_trial_ls.tsv")),
            "multils2024_en_trial_lcp": len(load_multils_en_lcp(BENCH / "multils2024_en_trial_lcp.tsv")),
            "note": "ranking eval (accuracy@k / MRR) needs a substitute candidate generator from the models side",
        },
        "notes": [
            f"C(w,s) via {scorer_desc}, with real sentence context.",
            "CWI binary label = >=1 of 20 annotators marked complex (lenient); "
            "threshold tuned on train rather than using the sigmoid midpoint 0.5.",
        ],
    }
    out_path = HERE / "results" / "detection_baseline.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"threshold (tuned on train): {best_thr}")
    for split in ("train", "dev"):
        m = out[split]
        print(f"  {split}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
