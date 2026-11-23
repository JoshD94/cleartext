"""Can blind ratings train a passive gate? (development check, November 22, 2026)

    python rated_gate_check.py STATE_DIR

Collects every rated passive item of rounds 1-4 whose output passive_v4
reproduces exactly (so the mark applies to v4's output), fits a standardized
logistic regression on the converter's candidate features plus five sentence
features, and reports (a) fit on rounds 1-3, scored on round 4 and (b) 5-fold
CV over all rows. Writes rated-gate.json here (aggregates only).
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROUNDS = [("rating-r1", "pool-Awide.json", ("passive", "chhoeng_agentive")),
          ("rating-r2", "pool-Bwide.json", ("passive", "chhoeng_agentive")),
          ("rating-r3", "pool-Cwide.json", ("passive_v3",)), ("rating-r4", "pool-Dwide.json", ("passive_v4",))]


def rows(state):
    from cleartext.structure_v2.common import parse
    from cleartext.structure_v2.passive import ActiveVoice
    from evals.structure_v2 import majority
    av, out = ActiveVoice(version=4), []
    for name, poolf, arms in ROUNDS:
        key = json.loads((state / f"{name}-key.json").read_text())
        pool = {r["id"]: r for r in json.loads((state / poolf).read_text())}
        marks = [dict(l.split(",")[:2] for l in (state / name / f"rater{r}" / "marks.csv").read_text().splitlines()[1:])
                 for r in (1, 2, 3)]
        seen = set()
        for it in key["items"]:
            if it["arm"] not in arms or (it["before"], it["after"]) in seen:
                continue
            seen.add((it["before"], it["after"]))
            new, edits = av.rewrite_sentence(it["before"], [parse(x) for x in pool[it["id"]]["context"]])
            if new == it["after"] and edits:
                out.append({"round": name, "before": it["before"], "features": [e.features for e in edits],
                            "label": int(majority([m[it["card"]] for m in marks]) == "good")})
    return out


def extra(row):
    from cleartext.structure_v2.common import parse
    toks = list(list(parse(row["before"]).sents)[0])
    by = [t for t in toks if t.lower_ == "by"]
    after = [t for t in toks if by and t.i > by[0].i]
    return [int(any(t.tag_ == "VBG" and t.dep_ in ("advcl", "acl", "xcomp")
                    and not any(c.dep_.startswith("nsubj") for c in t.children) for t in after)),
            int(any(t.lower_ == "only" for t in toks)), sum(t.text == "," for t in toks),
            int(any(t.like_num for t in toks)), len(row["features"])]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from cleartext.structure_v2.gate import vectorize
    data = rows(Path(sys.argv[1]))
    X = np.array([vectorize(r["features"][0]) + extra(r) for r in data])
    y = np.array([r["label"] for r in data])
    rd = np.array([r["round"] for r in data])
    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
    test = rd == "rating-r4"
    p = model().fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[b] = model().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    keep = {str(k): float(y[test][np.argsort(-p)[: int(round(k * test.sum()))]].mean()) for k in (0.5, 0.7, 0.85)}
    out = {"rows": len(y), "good": int(y.sum()), "by_round": {f"{k[0]}:{k[1]}": v for k, v in Counter(zip(rd, y)).items()},
           "fit_r1_r3_score_r4_auc": float(roc_auc_score(y[test], p)), "r4_good_rate_when_keeping": keep,
           "r4_base_good_rate": float(y[test].mean()), "cv5_all_rows_auc": float(roc_auc_score(y, oof))}
    (HERE / "rated-gate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
