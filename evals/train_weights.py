#!/usr/bin/env python3
"""Train ClearText eval weights from human judgments.

Implements Kea-Roy's Week 3 task: "Get trained weights for eval based on
human-rated simplicity data (make sure as close as possible to human evals).
Combine datasets?"

Built on evaluation/evaluate.py from PR #1 (vendored as evaluate_base.py),
which implements the Week 1 formulas from the "Kea-Roy W1 Formula" doc tab:

  Word:     C(w,s) = sigmoid(A + B*(-zipf(w)) + C*len(w) + D*syll(w)
                             + E*ctx(w,s) + F*pos(w))
  Sentence: S(x)   = G*S_lex + H*S_read + I*S_jargon   (G+H+I = 1)

Stage A: fit A-E on CompLex single-word human complexity ratings
         (SemEval-2021 Task 1, LCP single-word train/trial).
         F (POS) is untrainable here (no POS labels) and keeps PR #1's default.
Stage B: fit G/H/I on Lens SimpEval human simplicity ratings:
           - simpDA_2022.csv:   absolute simplicity of the simplified (0-100)
           - simplikert_2022.csv: relative simplicity change, Likert -2..+2
         Same 360 (original, simplified) pairs rated two ways -> the
         "combine datasets" experiment pools both views.

Outputs: trained_weights.json + prints a comparison report.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from wordfreq import zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_base import (  # noqa: E402
    clamp,
    jargon_simplicity,
    lexical_simplicity,
    readability_score,
    sigmoid,
    syllables,
    words,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent  # datasets live in ~/workspace/cleartext/

PR1_DEFAULT_WORD_W = {"A": 0.0, "B": 1.0, "C": 0.08, "D": 0.25, "E": 0.2, "F": 0.1}
PR1_DEFAULT_SENT_W = {"G": 1 / 3, "H": 1 / 3, "I": 1 / 3}


# --------------------------------------------------------------------------
# Shared feature helpers (same definitions as PR #1, but with real context)
# --------------------------------------------------------------------------

def context_complexity(token: str, sentence_words: list[str]) -> float:
    others = [w for w in sentence_words if w != token]
    if not others:
        return 0.0
    return sum(len(w) + syllables(w) for w in others) / len(others)


def word_features(token: str, sentence_words: list[str]) -> list[float]:
    """[-zipf, length, syllables, context_complexity, pos] for B..F."""
    return [
        -zipf_frequency(token, "en"),
        float(len(token)),
        float(syllables(token)),
        context_complexity(token, sentence_words),
        0.0,  # pos: no POS labels in training data; F stays at default
    ]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------
# Stage A: word weights A-E on CompLex
# --------------------------------------------------------------------------

def load_complex(path: Path):
    # NOTE: naive tab-split, not csv: sentences contain unbalanced double
    # quotes (e.g. bible corpus), which corrupts csv quote parsing and merges
    # rows (~430 train rows lost/misaligned with DictReader).
    rows = []
    with path.open(encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            assert len(fields) == 5, f"unexpected field count in {path}: {line[:80]!r}"
            _id, _corpus, sentence, token, complexity = fields
            rows.append((sentence, token, float(complexity)))
    return rows


def stage_a_word_weights():
    print("=" * 70)
    print("STAGE A: word-level weights A-E on CompLex (human complexity ratings)")
    print("=" * 70)
    train_rows = load_complex(DATA / "complex_single_train.tsv")
    trial_rows = load_complex(DATA / "complex_single_trial.tsv")
    print(f"train rows: {len(train_rows)}, trial rows: {len(trial_rows)}")

    def featurize(rows):
        X, y = [], []
        for sent, token, complexity in rows:
            sw = words(sent)
            X.append(word_features(token, sw)[:4])  # B..E (no POS labels)
            y.append(complexity)
        return np.array(X), np.array(y)

    X_train, y_train = featurize(train_rows)
    X_trial, y_trial = featurize(trial_rows)

    reg = LinearRegression().fit(X_train, y_train)
    trained = {
        "A": float(reg.intercept_),
        "B": float(reg.coef_[0]),
        "C": float(reg.coef_[1]),
        "D": float(reg.coef_[2]),
        "E": float(reg.coef_[3]),
        "F": PR1_DEFAULT_WORD_W["F"],  # untrained: no POS labels in CompLex
    }
    print(f"train R^2 (linear): {reg.score(X_train, y_train):.4f}")

    def trial_corr(w):
        preds = [sigmoid(w["A"] + w["B"] * r[0] + w["C"] * r[1]
                         + w["D"] * r[2] + w["E"] * r[3]) for r in X_trial]
        return pearson(preds, y_trial)

    r_trained = trial_corr(trained)
    r_default = trial_corr(PR1_DEFAULT_WORD_W)
    print(f"trial Pearson r vs human complexity:")
    print(f"  trained A-E : {r_trained:.4f}")
    print(f"  PR#1 default: {r_default:.4f}")
    print(f"  trained weights: " +
          ", ".join(f"{k}={v:.4f}" for k, v in trained.items()))
    return trained, {"trial_pearson_trained": r_trained,
                     "trial_pearson_default": r_default}


# --------------------------------------------------------------------------
# Stage B: sentence weights G/H/I on SimpEval
# --------------------------------------------------------------------------

def load_simpeval():
    """Return list of dicts: id, original, simplified, system, target_0_1, source."""
    pairs = {}

    def add(path, source):
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["Input.id"], r["Input.original"], r["Input.simplified"],
                       r["Input.system"])
                if source == "simpDA":
                    target = float(r["Answer.simplicity"]) / 100.0
                else:  # simplikert: Likert -2..+2 relative simplicity change
                    target = (float(r["Answer.simplicity"]) + 2.0) / 4.0
                pairs.setdefault((source, key), []).append(target)

    add(DATA / "simpDA_2022.csv", "simpDA")
    add(DATA / "simplikert_2022.csv", "simplikert")
    rows = []
    for (source, (pid, orig, simp, system)), targets in pairs.items():
        rows.append({
            "id": pid, "original": orig, "simplified": simp, "system": system,
            "target": sum(targets) / len(targets), "source": source,
        })
    return rows


def sentence_features(original: str, simplified: str, word_w: dict) -> dict:
    orig_words, simp_words = words(original), words(simplified)
    # S_lex: shared threshold-based definition (evaluate_base.lexical_simplicity):
    # normalized reduction in the count of words with C(w,s) > 0.5.
    s_lex = lexical_simplicity(orig_words, simp_words, word_w)
    s_read = clamp(readability_score(simplified) - readability_score(original) + 0.5)
    # S_jargon: shared formula (evaluate_base.jargon_simplicity). The "jargon
    # detector" (technical-term set) is a zipf<3.0 rarity proxy here because
    # SimpEval's news domain has no curated term list; the deployed formula
    # takes an explicit per-domain set instead.
    technical_terms = {w for w in orig_words if zipf_frequency(w, "en") < 3.0}
    s_jargon = jargon_simplicity(orig_words, simp_words, technical_terms)
    return {"S_lex": s_lex, "S_read": s_read, "S_jargon": s_jargon}


def s_score(feats: dict, w: dict) -> float:
    return w["G"] * feats["S_lex"] + w["H"] * feats["S_read"] + w["I"] * feats["S_jargon"]


def fit_simplex(X: np.ndarray, y: np.ndarray, objective: str = "corr",
                starts: list | None = None) -> dict:
    """Fit G/H/I >= 0, summing to 1. Objective: 'corr' (max Pearson r) or 'mse'."""
    cons = {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}
    bounds = [(0.0, 1.0)] * 3
    if starts is None:
        starts = [[1 / 3, 1 / 3, 1 / 3], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]

    def loss(w):
        preds = X @ w
        if objective == "corr":
            return -pearson(preds, y)
        return float(np.mean((preds - y) ** 2))

    best, best_loss = None, math.inf
    for s0 in starts:
        res = minimize(loss, np.array(s0), method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if res.fun < best_loss:
            best, best_loss = res.x, res.fun
    w = np.clip(best, 0, 1)
    w = w / w.sum()
    return {"G": float(w[0]), "H": float(w[1]), "I": float(w[2])}


def stage_b_sentence_weights(word_w=None, feature_fn=None):
    # feature_fn(original, simplified) -> {"S_lex","S_read","S_jargon"}.
    # Defaults to the linear formula with trained word weights; pass a GBM-based
    # feature_fn (see train_word_model.py) to refit with the non-linear word model.
    if feature_fn is None:
        if word_w is None:
            raise ValueError("stage_b needs word_w or feature_fn")
        def feature_fn(original, simplified):
            return sentence_features(original, simplified, word_w)
    print()
    print("=" * 70)
    print("STAGE B: sentence weights G/H/I on SimpEval human simplicity ratings")
    print("=" * 70)
    rows = load_simpeval()
    n_da = sum(1 for r in rows if r["source"] == "simpDA")
    n_lik = sum(1 for r in rows if r["source"] == "simplikert")
    print(f"pairs: {n_da} simpDA (absolute 0-100) + {n_lik} simplikert (relative -2..+2)")

    print("computing S_lex/S_read/S_jargon per pair...")
    feats, targets, ids, sources = [], [], [], []
    for r in rows:
        f = feature_fn(r["original"], r["simplified"])
        feats.append([f["S_lex"], f["S_read"], f["S_jargon"]])
        targets.append(r["target"])
        ids.append(r["id"])
        sources.append(r["source"])
    X = np.array(feats)
    y = np.array(targets)
    ids = np.array(ids)

    # split by original-sentence id: 48 train / 12 test (no pair leakage)
    rng = np.random.default_rng(42)
    uniq = sorted(set(ids))
    rng.shuffle(uniq)
    test_ids = set(uniq[:12])
    is_test = np.array([i in test_ids for i in ids])
    is_da = np.array([s == "simpDA" for s in sources])

    tr = ~is_test & is_da          # train: simpDA only
    tr_comb = ~is_test             # train: simpDA + simplikert (combined)
    te = is_test & is_da           # test: held-out simpDA sentence ids
    print(f"train pairs (simpDA): {tr.sum()}, train pairs (combined): {tr_comb.sum()}, "
          f"test pairs: {te.sum()}")

    w_corr = fit_simplex(X[tr], y[tr], "corr")
    w_mse = fit_simplex(X[tr], y[tr], "mse")
    w_corr_comb = fit_simplex(X[tr_comb], y[tr_comb], "corr")

    def ev(w, mask):
        return pearson(X[mask] @ np.array([w["G"], w["H"], w["I"]]), y[mask])

    print("\nPearson r vs human simplicity (train simpDA / held-out simpDA test):")
    header = f"  {'weights':28s} {'train':>7s} {'test':>7s}"
    print(header)
    cands = [
        ("trained (max corr)", w_corr),
        ("trained (least squares)", w_mse),
        ("trained (max corr, combined)", w_corr_comb),
        ("PR#1 default (1/3,1/3,1/3)", PR1_DEFAULT_SENT_W),
        ("S_lex only", {"G": 1.0, "H": 0.0, "I": 0.0}),
        ("S_read only", {"G": 0.0, "H": 1.0, "I": 0.0}),
        ("S_jargon only", {"G": 0.0, "H": 0.0, "I": 1.0}),
    ]
    results = {}
    for name, w in cands:
        r_tr, r_te = ev(w, tr), ev(w, te)
        results[name] = {"G": w["G"], "H": w["H"], "I": w["I"],
                         "train_r": r_tr, "test_r": r_te}
        print(f"  {name:28s} {r_tr:7.4f} {r_te:7.4f}")
    print("\n  fitted weights (max-corr, simpDA): " +
          ", ".join(f"{k}={v:.3f}" for k, v in w_corr.items()))
    print("  fitted weights (max-corr, combined): " +
          ", ".join(f"{k}={v:.3f}" for k, v in w_corr_comb.items()))

    # sanity: does the formula rank systems the way humans do?
    print("\nsystem ranking by human simplicity vs by trained S (test pairs):")
    systems = sorted(set(r["system"] for r in rows))
    wv = np.array([w_corr["G"], w_corr["H"], w_corr["I"]])
    hum, mod = {}, {}
    for s in systems:
        m = te & np.array([r["system"] == s for r in rows])
        if m.sum():
            hum[s] = float(y[m].mean())
            mod[s] = float((X[m] @ wv).mean())
    hrank = sorted(hum, key=hum.get, reverse=True)
    mrank = sorted(mod, key=mod.get, reverse=True)
    for s in hrank:
        mark = "✓" if hrank.index(s) == mrank.index(s) else " "
        print(f"  {mark} human #{hrank.index(s)+1}: {s:22s} "
              f"model #{mrank.index(s)+1}")
    return w_corr, w_corr_comb, results


def main() -> int:
    word_w, rep_a = stage_a_word_weights()
    sent_w, sent_w_comb, rep_b = stage_b_sentence_weights(word_w)

    out = {
        "word_weights": word_w,
        "sentence_weights": sent_w,  # recommended: max-corr fit on simpDA
        "sentence_weights_maxcorr_combined": sent_w_comb,
        "defaults": {"word": PR1_DEFAULT_WORD_W, "sentence": PR1_DEFAULT_SENT_W},
        "stage_a": rep_a,
        "stage_b": rep_b,
        "notes": [
            "RECOMMENDED: use trained A-E for word complexity (trial Pearson r 0.68 vs 0.57 "
            "for PR#1 defaults on held-out CompLex trial).",
            "RECOMMENDED: use fitted G/H/I (max-corr on simpDA) instead of the 1/3 "
            "defaults: held-out r 0.63 vs 0.38. The 1/3 default underperforms now "
            "because S_jargon (zipf<3.0 proxy on news text) is near-noise here; the "
            "fit correctly downweights it (I~0.04). Revisit I with a real per-domain "
            "jargon term list.",
            "S_lex is threshold-based since 2026-10-04: normalized reduction in the "
            "count of words with C(w,s) > 0.5 (was: mean-complexity difference). "
            "This raised S_lex alone from test r=0.05 to r=0.44.",
            "S_jargon uses the same normalized-reduction form over a technical-term "
            "set; training uses a zipf<3.0 rarity proxy for the term set on "
            "SimpEval's news domain.",
            "'Combine datasets' experiment: pooling simpDA_2022 (absolute 0-100) with "
            "simplikert_2022 (relative -2..+2) views of the same 360 pairs gives "
            "near-identical fitted weights (test r 0.62 vs 0.63) — redundant views.",
            "A-E fit by least squares on CompLex single-word train (7662 human complexity ratings).",
            "F (POS) kept at PR#1 default 0.1: no POS labels in training data.",
            "Word features use real sentence context for E (context complexity); "
            "PR #1's evaluate_record did the same after the 2026-10-04 E-term fix "
            "(it previously passed empty context, zeroing E).",
            "Targets: simpDA mean(Answer.simplicity)/100; simplikert mean((Answer.simplicity+2)/4).",
            "CompLex TSVs contain unbalanced double quotes; parsed with naive tab-split, not csv.",
        ],
    }
    out_path = HERE / "trained_weights.json"
    out_path.write_text(json.dumps(out, indent=2))
    print()
    print(f"wrote {out_path}")
    print("RECOMMENDED word weights (trained A-E): " +
          ", ".join(f"{k}={v:.4f}" for k, v in word_w.items()))
    print("RECOMMENDED sentence weights (max-corr fit): " +
          ", ".join(f"{k}={v:.3f}" for k, v in sent_w.items()))
    try:
        from build_status import build as build_status_page
        build_status_page()
        print("rebuilt evals/status.html")
    except Exception as exc:  # never break training over the status page
        print(f"(status page rebuild skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
