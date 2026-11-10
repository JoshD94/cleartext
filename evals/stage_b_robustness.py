#!/usr/bin/env python3
"""Stage B robustness: repeated random splits instead of one 48/12 split.

The headline Stage B result (fitted G/H/I test r=0.60 vs 0.37 defaults) rests
on a single random 48-train/12-test split of 60 sentence ids -- thin ice.
This script repeats the split 25 times (seeded) and reports mean +/- std of
held-out test Pearson r for each candidate weighting, plus the distribution
of fitted G/H/I. If the fitted weights beat defaults across splits, the
recommendation is robust; if the intervals overlap heavily, it isn't.

Writes results/stage_b_robustness.json (read live by build_status.py).

Usage: ./venv/bin/python evals/stage_b_robustness.py [--reps 25]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from train_weights import (  # noqa: E402
    PR1_DEFAULT_SENT_W,
    fit_simplex,
    load_simpeval,
    pearson,
    sentence_features,
)


def trained_word_weights():
    return json.loads((HERE / "trained_weights.json").read_text())["word_weights"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    args = ap.parse_args()

    word_w = trained_word_weights()
    rows = load_simpeval()
    rows = [r for r in rows if r["source"] == "simpDA"]
    print(f"simpDA pairs: {len(rows)}")

    print("computing S_lex/S_read/S_jargon per pair (linear pipeline)...")
    X = np.array([
        [f["S_lex"], f["S_read"], f["S_jargon"]]
        for f in (sentence_features(r["original"], r["simplified"], word_w)
                  for r in rows)
    ])
    y = np.array([r["target"] for r in rows])
    ids = np.array([r["id"] for r in rows])
    uniq = sorted(set(ids))
    assert len(uniq) == 60, f"expected 60 sentence ids, got {len(uniq)}"

    cands = {
        "fitted (max corr)": None,  # fit per rep
        "PR#1 default (1/3)": PR1_DEFAULT_SENT_W,
        "S_read only": {"G": 0.0, "H": 1.0, "I": 0.0},
        "S_lex only": {"G": 1.0, "H": 0.0, "I": 0.0},
    }
    test_rs = {k: [] for k in cands}
    fitted_w = []

    for rep in range(args.reps):
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(uniq)
        test_ids = set(perm[:12])
        is_test = np.array([i in test_ids for i in ids])
        tr, te = ~is_test, is_test
        w_fit = fit_simplex(X[tr], y[tr], "corr")
        fitted_w.append([w_fit["G"], w_fit["H"], w_fit["I"]])
        for name, w in cands.items():
            wv = w_fit if w is None else w
            r = pearson(X[te] @ np.array([wv["G"], wv["H"], wv["I"]]), y[te])
            test_rs[name].append(float(r))

    fitted_w = np.array(fitted_w)
    summary = {}
    print(f"\nheld-out test Pearson r over {args.reps} random 48/12 splits:")
    for name, rs in test_rs.items():
        rs = np.array(rs)
        summary[name] = {
            "mean": round(float(rs.mean()), 4),
            "std": round(float(rs.std()), 4),
            "min": round(float(rs.min()), 4),
            "max": round(float(rs.max()), 4),
        }
        print(f"  {name:22s} {rs.mean():.3f} +/- {rs.std():.3f}  "
              f"[{rs.min():.3f}, {rs.max():.3f}]")
    fw_mean, fw_std = fitted_w.mean(0), fitted_w.std(0)
    print(f"\nfitted G/H/I across reps: "
          f"G={fw_mean[0]:.3f}+/-{fw_std[0]:.3f} "
          f"H={fw_mean[1]:.3f}+/-{fw_std[1]:.3f} "
          f"I={fw_mean[2]:.3f}+/-{fw_std[2]:.3f}")

    out = {
        "reps": args.reps,
        "test_r": summary,
        "fitted_weights": {
            k: {"mean": round(float(m), 4), "std": round(float(s), 4)}
            for k, m, s in zip(("G", "H", "I"), fw_mean, fw_std)
        },
        "notes": [
            "Repeated 48-train/12-test splits over the 60 simpDA sentence ids.",
            "Features from the recommended linear pipeline (trained A-E).",
        ],
    }
    path = HERE / "results" / "stage_b_robustness.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
