"""Round 4: a strict meaning checker and a target-difficulty floor.

  learned-words strict fit

Strict checker: the same features and model family as the stage-4 checker,
trained only on rated SWORDS-dev substitutes (>= 3 ratings, window NLI
present), with label = accepted by >= 80% of annotators. The stage-4 checker
counted >= 50% as acceptable, so substitutes that shift nuance (crucial ->
important) were positives. Its threshold (fixed before fitting): the
smallest value in 0.50, 0.51, ..., 0.99 whose out-of-fold SWORDS-dev
precision (strict label) is >= 0.90 with >= 30 rows. Tested once on rated
SWORDS-test rows. That rule found no threshold (first run, job 1afb0ba6b4e8);
fallback S2, fixed before any grid was seen: the threshold with the best
out-of-fold F0.5 on the strict label (>= 30 rows).

Target floor: a target is eligible only if WordDifficulty's CWI regressor
predicts that >= 10% of annotators (2 of 20) mark it complex. Stage 1 used
"at least one of 20". Fixed a priori, not fitted.

Neither reads Joshua's marks or any rated swap.
"""

import json
import pickle
import time

import numpy as np

from .learned_words import REPORT, STATE

STRICT = STATE / "strict-checker.pkl"
PRECOMMIT = REPORT / "strict-precommit.json"
STRICT_SHARE = 0.8
TARGET_FLOOR = 0.10


def _sha(path):
    import hashlib

    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def rated_rows(rows, ds):
    from .learned_words_study import has_nli

    return [r for r in rows if r["dataset"] == ds and r["rated"] is not None and r["rated_n"] >= 3 and has_nli(r)]


class StrictChecker:
    def fit(self, rows):
        from cleartext.learned_words.decide import check_keys, gbm, matrix, oof

        self.keys = check_keys(rows)
        X = matrix(rows, self.keys)
        y = np.array([float(r["rated"] >= STRICT_SHARE) for r in rows])
        g = np.array([r["group"] for r in rows])
        self.oof = oof(gbm, X, y, g)
        self.model = gbm().fit(X, y)
        return self

    def score(self, rows):
        from cleartext.learned_words.decide import matrix

        return self.model.predict_proba(matrix(rows, self.keys))[:, 1] if rows else np.array([])


def fit(argv):
    from sklearn.metrics import roc_auc_score

    from .learned_words_simpler import SIMPLER
    from .learned_words_study import MODEL, _rows, all_instances, decide, precision

    t0 = time.time()
    rows = _rows()
    stage = pickle.load(open(MODEL, "rb"))
    for r in rows:
        r["f"].update(stage["extra"].get(r["id"], {"p_complex": 0.5, "p_term": 0.5}))
    dev, test = rated_rows(rows, "swords_dev"), rated_rows(rows, "swords_test")
    sc = StrictChecker().fit(dev)
    ydev = np.array([r["rated"] >= STRICT_SHARE for r in dev])
    grid = []
    for t in np.round(np.arange(0.50, 0.995, 0.01), 2):
        s = sc.oof >= t
        grid.append({"t": float(t), "rows": int(s.sum()), "precision": float(ydev[s].mean()) if s.any() else None})
    npos = int(ydev.sum())
    for g in grid:
        s = sc.oof >= g["t"]
        tp = int(ydev[s].sum())
        p, r = (tp / s.sum() if s.any() else 0.0), tp / npos
        g["recall"], g["f05"] = r, (1.25 * p * r / (0.25 * p + r) if p + r else 0.0)
    ok = [g for g in grid if g["precision"] is not None and g["precision"] >= 0.90 and g["rows"] >= 30]
    if ok:
        sc.threshold, rule = ok[0]["t"], "precision >= 0.90"
    else:
        best = max((g for g in grid if g["rows"] >= 30), key=lambda g: (g["f05"], g["t"]))
        sc.threshold, rule = best["t"], "fallback S2: best OOF F0.5"
    print("STRICT RULE", rule, sc.threshold, flush=True)
    with open(STRICT, "wb") as f:
        pickle.dump(sc, f)

    yt = np.array([r["rated"] >= STRICT_SHARE for r in test])
    yl = np.array([r["rated"] >= 0.5 for r in test])
    ps, pl = sc.score(test), stage["stage"].check(test)
    s, l = ps >= sc.threshold, pl >= stage["stage"].tc

    def prec(sel, y):
        return {"accepted": int(sel.sum()), "precision": float(y[sel].mean()) if sel.any() else None}
    swords = {"rows": len(test), "strict_positive_share": float(yt.mean()),
              "strict_auc": float(roc_auc_score(yt, ps)), "lenient_checker_auc_on_strict_label": float(roc_auc_score(yt, pl)),
              "strict_at_threshold": prec(s, yt), "lenient_at_tc_strict_label": prec(l, yt),
              "both": prec(s & l, yt), "both_lenient_label": prec(s & l, yl)}

    # Effect on the stage's public test decisions (strict checker on the top candidate).
    simpler = pickle.load(open(SIMPLER, "rb"))
    inst = all_instances()
    effect = {}
    for ds in ("tsar_test", "nnseval"):
        rs = [dict(r, f=dict(r["f"])) for r in rows if r["dataset"] == ds]
        dec = decide(stage, rs)
        top = {(r["id"], r["word"].lower()): r for r in rs}
        sw = {k: d for k, d in dec.items() if d["swap"]}
        trow = [top[(k, d["word"].lower())] for k, d in sw.items()]
        st = dict(zip(sw, sc.score(trow)))
        diff = dict(zip(sw, simpler.raw([d["target"] for d in sw.values()], "cwi")))
        sets = [i for i in inst if i["dataset"] == ds]
        res = {"stage": precision(dec, sets)}
        for name, keep in (("strict", lambda k: st[k] >= sc.threshold),
                           ("floor", lambda k: diff[k] >= TARGET_FLOOR),
                           ("strict_and_floor", lambda k: st[k] >= sc.threshold and diff[k] >= TARGET_FLOOR)):
            res[name] = precision({k: dict(d, swap=d["swap"] and (k not in sw or keep(k))) for k, d in dec.items()}, sets)
        effect[ds] = res
    tsar_targets = [i["target"] for i in inst if i["dataset"] == "tsar_test"]
    floor = {"tsar_test_targets_passing": float(np.mean(simpler.raw(tsar_targets, "cwi") >= TARGET_FLOOR))}
    out = {"rule": __doc__.strip(), "applied_rule": rule, "threshold": sc.threshold, "dev_rows": len(dev),
           "dev_strict_positive_share": float(ydev.mean()), "dev_oof_auc": float(roc_auc_score(ydev, sc.oof)),
           "grid": grid, "swords_test": swords, "stage_effect": effect, "target_floor": floor,
           "seconds": round(time.time() - t0, 1)}
    (REPORT / "strict-check.json").write_text(json.dumps(out, indent=1))
    PRECOMMIT.write_text(json.dumps({
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "strict_threshold": sc.threshold, "applied_rule": rule,
        "target_floor": TARGET_FLOOR, "strict_sha256": _sha(STRICT),
        "source_sha256": {p: _sha(p) for p in ("evals/learned_words_strict.py",)},
        "note": "written before any round-4 swap exists"}, indent=1))
    print(json.dumps({k: out[k] for k in ("threshold", "dev_rows", "dev_strict_positive_share", "dev_oof_auc",
                                          "swords_test", "stage_effect", "target_floor")}, indent=1))
    print("COMPLETE", flush=True)


def load():
    """(strict checker, target floor) after checking the pre-commitment."""
    pre = json.load(open(PRECOMMIT))
    if _sha(STRICT) != pre["strict_sha256"]:
        raise SystemExit("REFUSED: strict-checker.pkl differs from the pre-commitment")
    return pickle.load(open(STRICT, "rb")), pre["target_floor"]


def main(argv):
    if argv[:1] == ["fit"]:
        return fit(argv[1:])
    raise SystemExit("usage: learned-words strict fit")
