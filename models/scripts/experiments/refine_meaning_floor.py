"""Meaning floor: reject a candidate whose meaning match (probability that the target's sense is one the candidate
can express) is below a floor. v5 tables, out-of-fold decision probabilities (10-fold x3 grouped), BenchLS as dev;
TSAR report-only. Adopt only if dev net improves by >= 5 over no floor."""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import cleartext.refine as R
from cleartext.data import ROOT

names, S = R.tables("v5")
R.SPECIFICITY = True
SENSE = names.index("sense_fit")
dev = S["benchls_dev"] + S["benchls_holdout"]
tsar = S["tsar_test"]
X, y, g, k = R.flat_specific(dev)
Xt, _, _, kt = R.flat_specific(tsar)
oof = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    perm = dict(zip(ug, rng.permutation(len(ug)) % 10))
    fold = np.array([perm[x] for x in g])
    p = np.zeros(len(y))
    for f in range(10):
        p[fold == f] = (
            R.AverageDecision()
            .fit(X[fold != f], y[fold != f])
            .predict_proba(X[fold == f])[:, 1]
        )
    oof.append(dict(zip(k, p)))
Pt = dict(zip(kt, R.AverageDecision().fit(X, y).predict_proba(Xt)[:, 1]))


def rule(P, floor, t=0.40):
    def choose(case):
        ok = [
            c
            for c in case["candidates"]
            if c["guard"] and P[case["id"], c["word"]] >= t and c["x"][SENSE] >= floor
        ]
        return max(ok, key=lambda c: P[case["id"], c["word"]]) if ok else None

    return choose


out = {}
for floor in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
    nets = [R.evaluate(dev, rule(P, floor))["net"] for P in oof]
    r = R.evaluate(tsar, rule(Pt, floor))
    out[floor] = {"dev_nets": nets, "tsar": r}
    print(
        f"floor {floor:.2f}: dev net {np.mean(nets):6.1f} {nets} | tsar {r['correct']}/{r['edits']} net {r['net']}",
        flush=True,
    )
blocked = sum(c["x"][SENSE] < 0.2 for case in dev for c in case["candidates"])
total = sum(len(case["candidates"]) for case in dev)
print(f"candidates under 0.20 meaning match: {blocked} of {total}")
json.dump(
    {str(k): v for k, v in out.items()},
    open(ROOT / "runs/ensemble-v5-20260927/refine_meaning_floor.json", "w"),
    indent=2,
)
