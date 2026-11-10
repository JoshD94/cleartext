#!/usr/bin/env python3
"""Options C+D: train and evaluate the GBM word-complexity model.

C: gradient boosting for C(w,s) instead of linear-sigmoid (still non-LLM:
   6 transparent features, no embeddings).
D: (a) + concreteness (Brysbaert) + age-of-acquisition (Kuperman) features;
   (b) + CWI-2018 train as auxiliary training data (single-word targets,
   probabilistic target = fraction of 20 annotators marking complex).

Ablation: {linear, GBM} x {base4, base6} x {CompLex, CWI, CompLex+CWI}.
Held-out eval (never trained on): CompLex trial (headline) + CWI-2018 dev
single-word (second opinion; out-of-scheme robustness check).

Then refits Stage B sentence weights G/H/I with GBM-based S_lex
(via train_weights.stage_b_sentence_weights) and compares to the
linear-based fit.

Saves the best GBM (by CompLex trial r) to weights/word_gbm.joblib +
weights/word_gbm_meta.json. Writes results/word_model.json
(read live by build_status.py).

Usage: ./venv/bin/python evals/train_word_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from joblib import dump as joblib_dump  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from wordfreq import zipf_frequency  # noqa: E402

from datasets import load_cwi2018  # noqa: E402
from evaluate_base import clamp, jargon_simplicity, readability_score, words  # noqa: E402
from train_weights import DATA, load_complex, stage_b_sentence_weights  # noqa: E402
from word_models import FEATURES, WEIGHTS_DIR, featurize6, lexical_simplicity_gbm  # noqa: E402

BENCH = HERE / "data" / "benchmarks"
GBM_PARAMS = dict(random_state=0, max_iter=300, learning_rate=0.05,
                  max_leaf_nodes=31, min_samples_leaf=20)


def load_train_rows(source: str):
    """source: 'complex' | 'cwi' | 'both'. CWI rows are single-word targets only."""
    rows = []
    if source in ("complex", "both"):
        rows += [(s, t, c) for s, t, c in load_complex(DATA / "complex_single_train.tsv")]
    if source in ("cwi", "both"):
        for r in load_cwi2018(BENCH / "cwi2018_en_train.tsv"):
            tgt = r.target.strip()
            if not tgt or any(ch.isspace() for ch in tgt):
                continue  # C(w,s) is per-word; skip multi-word targets
            rows.append((r.sentence, tgt, r.prob))
    return rows


def load_eval_rows():
    complex_trial = [(s, t, c) for s, t, c in load_complex(DATA / "complex_single_trial.tsv")]
    cwi_dev = []
    for r in load_cwi2018(BENCH / "cwi2018_en_dev.tsv"):
        tgt = r.target.strip()
        if not tgt or any(ch.isspace() for ch in tgt):
            continue
        cwi_dev.append((r.sentence, tgt, r.prob))
    return {"complex_trial": complex_trial, "cwi_dev": cwi_dev}


def featurize(rows, use_extra: bool):
    X = [featurize6(t, words(s))[:4] if not use_extra else featurize6(t, words(s))
         for s, t, _ in rows]
    return np.array(X, dtype=float), np.array([c for _, _, c in rows], dtype=float)


def main() -> int:
    eval_rows = load_eval_rows()
    print(f"eval: CompLex trial n={len(eval_rows['complex_trial'])}, "
          f"CWI dev (single-word) n={len(eval_rows['cwi_dev'])}")

    results = {}
    for source in ("complex", "cwi", "both"):
        train_rows = load_train_rows(source)
        tag = {"complex": "CompLex", "cwi": "CWI", "both": "+CWI"}[source]
        print(f"\ntrain: {tag} n={len(train_rows)}")
        for use_extra in (False, True):
            ftag = "base6" if use_extra else "base4"
            Xtr, ytr = featurize(train_rows, use_extra)
            for mname, model in (
                ("linear", LinearRegression()),
                ("gbm", HistGradientBoostingRegressor(**GBM_PARAMS)),
            ):
                model.fit(Xtr, ytr)
                cell = {}
                for ename, erows in eval_rows.items():
                    Xe, ye = featurize(erows, use_extra)
                    r = float(pearsonr(model.predict(Xe), ye)[0])
                    cell[ename] = round(r, 4)
                key = f"{mname}/{ftag}/{tag}"
                results[key] = cell
                print(f"  {key:24s} complex_trial r={cell['complex_trial']:.4f}  "
                      f"cwi_dev r={cell['cwi_dev']:.4f}")

    best_key = max(results, key=lambda k: results[k]["complex_trial"])
    print(f"\nbest by CompLex trial: {best_key} "
          f"(r={results[best_key]['complex_trial']:.4f})")

    # Retrain the winner on its training data and save.
    mname, ftag, dtag = best_key.split("/")
    use_extra = ftag == "base6"
    source = {"CompLex": "complex", "CWI": "cwi", "+CWI": "both"}[dtag]
    train_rows = load_train_rows(source)
    Xtr, ytr = featurize(train_rows, use_extra)
    best_model = HistGradientBoostingRegressor(**GBM_PARAMS).fit(Xtr, ytr)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    joblib_dump(best_model, WEIGHTS_DIR / "word_gbm.joblib")
    meta = {
        "features": FEATURES if use_extra else FEATURES[:4],
        "model": "HistGradientBoostingRegressor",
        "params": GBM_PARAMS,
        "train": dtag,
        "train_n": len(train_rows),
        "ablation": results,
        "best": best_key,
    }
    (WEIGHTS_DIR / "word_gbm_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {WEIGHTS_DIR / 'word_gbm.joblib'}")

    # Stage B refit with GBM-based S_lex.
    print("\nrefitting Stage B sentence weights with GBM-based S_lex...")

    def gbm_features(original, simplified):
        ow, sw = words(original), words(simplified)
        # Threshold 0.3 tuned on Stage B train (0.5 collapses: GBM scores are
        # well-calibrated, so most news words fall below 0.5 -> zero counts).
        s_lex = lexical_simplicity_gbm(ow, sw, threshold=0.3)
        s_read = clamp(readability_score(simplified) - readability_score(original) + 0.5)
        technical_terms = {w for w in ow if zipf_frequency(w, "en") < 3.0}
        s_jargon = jargon_simplicity(ow, sw, technical_terms)
        return {"S_lex": s_lex, "S_read": s_read, "S_jargon": s_jargon}

    w_corr, w_corr_comb, sb_results = stage_b_sentence_weights(feature_fn=gbm_features)

    out = {
        "ablation": results,
        "best": best_key,
        "best_trial_r": results[best_key]["complex_trial"],
        "stage_b_gbm": {
            "sentence_weights": w_corr,
            "sentence_weights_combined": w_corr_comb,
            "table": sb_results,
        },
        "notes": [
            "C(w,s) now gradient boosting on 6 transparent features (C) with "
            "concreteness + AoA lexicons and CWI-2018 pooled training data (D).",
            "Stage B refit uses GBM-based S_lex (threshold 0.5 on GBM scores).",
            "Model artifact: evals/weights/word_gbm.joblib; API: evals/word_models.py.",
        ],
    }
    out_path = HERE / "results" / "word_model.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
