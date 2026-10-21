"""Decision with BERT members: v5 tables versus v7 tables (BERT meaning mix and slot fit), same decision model,
10-fold x3 grouped CV on BenchLS, meaning floor 0.10 in both. TSAR report-only. Noise bar: net +5 or log-loss -0.001."""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from sklearn.metrics import log_loss
import cleartext.refine as R
from cleartext.data import ROOT

R.SPECIFICITY = True
out = {}
for version in ["v5", "v7"]:
    names, S = R.tables(version)
    SENSE = names.index("sense_fit")
    dev = S["benchls_dev"] + S["benchls_holdout"]
    tsar = S["tsar_test"]
    X, y, g, k = R.flat_specific(dev)
    Xt, _, _, kt = R.flat_specific(tsar)

    def rule(P, t, floor=0.10):
        def choose(case):
            ok = [
                c
                for c in case["candidates"]
                if c["guard"]
                and P[case["id"], c["word"]] >= t
                and c["x"][SENSE] >= floor
            ]
            return max(ok, key=lambda c: P[case["id"], c["word"]]) if ok else None

        return choose

    nets = []
    losses = []
    ts = []
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
        P = dict(zip(k, p))
        grid = {
            t: R.evaluate(dev, rule(P, t))["net"]
            for t in np.round(np.arange(0.25, 0.65, 0.05), 2)
        }
        t = max(grid, key=grid.get)
        nets.append(grid[t])
        ts.append(t)
        losses.append(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    t = float(np.median(ts))
    r = R.evaluate(
        tsar,
        rule(dict(zip(kt, R.AverageDecision().fit(X, y).predict_proba(Xt)[:, 1])), t),
    )
    out[version] = {
        "dev_nets": nets,
        "dev_log_loss": float(np.mean(losses)),
        "threshold": t,
        "tsar": r,
        "columns": len(names),
    }
    print(
        f"{version}: dev net {np.mean(nets):6.1f} {nets} log-loss {np.mean(losses):.4f} t={t} | tsar {r['correct']}/{r['edits']} net {r['net']}",
        flush=True,
    )
json.dump(
    out,
    open(ROOT / "runs/ensemble-v7-20260928/refine_bert_decision.json", "w"),
    indent=2,
)
