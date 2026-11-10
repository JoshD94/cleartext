"""Adjective reading for predicate participles: v5 tables (tagger reading) versus v5adj tables (built after the
generation.predicate_adjective change). Same decision model and protocol; meaning floor 0.10 in both."""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import cleartext.refine as R
from cleartext.data import ROOT

R.SPECIFICITY = True
out = {}
for version in ["v5", "v5adj"]:
    names, S = R.tables(version)
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

    def rule(P, t=0.40, floor=0.10):
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

    nets = [R.evaluate(dev, rule(P))["net"] for P in oof]
    r = R.evaluate(tsar, rule(Pt))
    changed = sum(
        1
        for c in dev
        if (c.get("target_info") or (0, "x"))[1] == "a"
        and c["target"].lower().endswith("ed")
    )
    out[version] = {"dev_nets": nets, "tsar": r}
    print(
        f"{version}: dev net {np.mean(nets):6.1f} {nets} | tsar {r['correct']}/{r['edits']} net {r['net']} | -ed adjective targets in dev {changed}",
        flush=True,
    )
json.dump(
    out,
    open(ROOT / "runs/ensemble-v5-20260927/refine_adjective_reading.json", "w"),
    indent=2,
)
