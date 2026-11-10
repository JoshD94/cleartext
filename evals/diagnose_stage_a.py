#!/usr/bin/env python3
"""Stage A diagnostics: human-agreement ceiling + model-class comparison.

1. CEILING: CompLex ships only averaged ratings (no per-annotator data), so the
   ceiling is estimated from CWI-2018 English, which records native (10 raters)
   and non-native (10 raters) complexity counts separately. Split-half Pearson
   between the two independent rater groups, plus the Spearman-Brown stepped-up
   estimate for the full 20-rater mean.
2. MODEL COMPARISON: same CompLex train/trial split and same 4 features
   (B..E) for three model classes -- current least-squares linear weights,
   RidgeCV, and HistGradientBoostingRegressor (non-linear diagnostic).

Writes results/stage_a_diagnostics.json (read live by build_status.py).

Usage: ./venv/bin/python evals/diagnose_stage_a.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402

from datasets import load_cwi2018  # noqa: E402
from train_weights import load_complex, word_features, DATA  # noqa: E402


def ceiling_cwi() -> dict:
    rows = load_cwi2018(HERE / "data" / "benchmarks" / "cwi2018_en_train.tsv")
    # NOTE: load_cwi2018 keeps only binary/prob; reload counts here.
    native, nonnative = [], []
    for line in (HERE / "data" / "benchmarks" / "cwi2018_en_train.tsv").read_text(
        encoding="utf-8"
    ).splitlines():
        f = line.split("\t")
        if len(f) < 11:
            continue
        native.append(int(f[7]) / 10.0)
        nonnative.append(int(f[8]) / 10.0)
    r, _ = pearsonr(native, nonnative)
    r = float(r)
    return {
        "n": len(native),
        "split_half_pearson": round(r, 4),
        "spearman_brown_20raters": round(2 * r / (1 + r), 4),
        "method": "CWI-2018 EN train: mean complexity of 10 native vs 10 "
                  "non-native annotators per word (independent rater groups)",
    }


def featurize(rows):
    from evaluate_base import words

    X, y = [], []
    for sent, token, complexity in rows:
        X.append(word_features(token, words(sent))[:4])  # B..E, as in Stage A
        y.append(complexity)
    return np.array(X), np.array(y)


def main() -> int:
    out: dict = {}

    print("ceiling: CWI-2018 split-half human agreement...")
    out["ceiling"] = ceiling_cwi()
    print(f"  split-half r = {out['ceiling']['split_half_pearson']}, "
          f"Spearman-Brown(20 raters) = {out['ceiling']['spearman_brown_20raters']}")

    print("Stage A model comparison on CompLex (train -> trial)...")
    Xtr, ytr = featurize(load_complex(DATA / "complex_single_train.tsv"))
    Xte, yte = featurize(load_complex(DATA / "complex_single_trial.tsv"))

    models = {
        "linear_least_squares": LinearRegression(),
        "ridge_cv": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "gradient_boosting": HistGradientBoostingRegressor(random_state=0),
    }
    comp = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        tr_r = float(pearsonr(model.predict(Xtr), ytr)[0])
        te_r = float(pearsonr(model.predict(Xte), yte)[0])
        comp[name] = {"train_r": round(tr_r, 4), "trial_r": round(te_r, 4)}
        print(f"  {name:22s} train r={tr_r:.4f}  trial r={te_r:.4f}")
        if name == "gradient_boosting" and hasattr(model, "feature_importances_"):
            pass

    # feature importances from the GBM (permutation-free, impurity-based)
    gbm = models["gradient_boosting"]
    try:
        from sklearn.inspection import permutation_importance

        perm = permutation_importance(gbm, Xte, yte, n_repeats=5, random_state=0)
        comp["gradient_boosting"]["permutation_importance"] = {
            k: round(float(v), 4)
            for k, v in zip(
                ["neg_zipf", "word_len", "syllables", "context"], perm.importances_mean
            )
        }
        print("  GBM permutation importance:",
              comp["gradient_boosting"]["permutation_importance"])
    except Exception as exc:
        print(f"  (permutation importance skipped: {exc})")

    out["model_comparison"] = comp
    out["notes"] = [
        "Same CompLex train (7662) / trial (421) split and same 4 features as Stage A.",
        "Ceiling estimated on CWI-2018 because CompLex ships only averaged ratings.",
    ]
    path = HERE / "results" / "stage_a_diagnostics.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
