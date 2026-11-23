"""Round 3: the "is the replacement simpler" check (cleartext.learned_words.simpler).

  learned-words simpler fit       train on CompLex train + CWI Train, choose the
                                  margin on unseen validation words (CompLex
                                  trial, CWI Dev) by the pre-registered rule,
                                  then score test words once; write simpler.pkl,
                                  simpler-check.json and simpler-precommit.json
  learned-words simpler round2    held-out comparison on the 98 round-2 swaps
                                  (Claude's marks; never used for fitting or
                                  the margin), after the pre-commitment exists

Rule (fixed before fitting): margin = smallest value in 0, 0.05, ..., 3 at
which >= 80% of validation word pairs with predicted margin >= it are
"clearly simpler" (observed gap >= 0.10 in the source's own units), with at
least 100 pairs, pooled over both sources with equal pair counts.

That rule FAILED its sanity check (margin 1.6 passes only 14% of TSAR-test
human simplifications and removes 28 of the learned stage's 34 correct TSAR
swaps): precision on random word pairs is the wrong target for substitution
pairs. Revised rule R2 (labelled separately; prompted by the TSAR-test pass
share, so TSAR-test numbers under R2 are not clean), set on train-side data
only, before the round-2 comparison ran:

  learned-words simpler margin-r2
      margin = the grid value (-1, -0.95, ..., 2) with the best balanced
      accuracy separating train-side human simplifications (BenchLS train
      and TSAR trial gold substitutes) from SWORDS-dev accepted substitutes
      (meaning-preserving synonyms not chosen for simplicity), ties to the
      higher margin. simpler.pkl is unchanged.
"""

import hashlib
import json
import pickle
import time
from collections import Counter

import numpy as np

from .learned_words import REPORT, STATE, read_jsonl

SIMPLER = STATE / "simpler.pkl"
PRECOMMIT = REPORT / "simpler-precommit.json"
PRECOMMIT_R2 = REPORT / "simpler-precommit-r2.json"
ROUND2_KEY = STATE / "rating/key.json"
ROUND2_MARKS = STATE / "rating/claude-marks.csv"


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _pairs_eval(model, data, part):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from cleartext.learned_words import simpler as S

    out, M, L = {}, [], []
    for src, d in sorted(data.items()):
        words = sorted(d[part])
        pred = dict(zip(words, model.difficulty(words)))
        m, lab, gap = S.pairs(d[part], pred, S.DELTA[src], max_pairs=100000)
        clear = np.abs(gap) >= S.DELTA[src]
        out[src] = {"words": len(words),
                    "spearman_vs_rating": float(spearmanr([d[part][w] for w in words],
                                                          [pred[w] for w in words])[0]),
                    "pair_auc_clear_pairs": float(roc_auc_score(gap[clear] > 0, m[clear]))}
        M.append(m), L.append(lab)
    return out, np.concatenate(M), np.concatenate(L)


def public_substitutions(model, margin):
    """Share of human substitutes that pass the margin, by data set (target -> substitute)."""
    from cleartext.learned_words import data as D

    res = {}
    sets = {"tsar_test_gold_2plus": [], "tsar_test_gold_1": [], "nnseval_gold": [],
            "swords_test_accepted": [], "swords_test_rejected": []}
    for i in D.tsar("test"):
        for w, c in i["gold"].items():
            if " " not in w:
                sets["tsar_test_gold_2plus" if c >= 2 else "tsar_test_gold_1"].append((i["target"], w))
    for i in D.ls_sets():
        if i["dataset"] == "nnseval":
            sets["nnseval_gold"] += [(i["target"], w) for w in i["gold"] if " " not in w]
    for i in D.swords("test"):
        for w, share, n in i["rated"]:
            if " " in w or n < 3:
                continue
            if share >= 0.5:
                sets["swords_test_accepted"].append((i["target"], w))
            elif share == 0:
                sets["swords_test_rejected"].append((i["target"], w))
    for k, prs in sets.items():
        prs = [(a, b) for a, b in prs if a.isalpha() and b.isalpha() and a.lower() != b.lower()]
        m = model.margin([a for a, _ in prs], [b for _, b in prs])
        res[k] = {"pairs": len(prs), "pass_share": float(np.mean(m >= margin)) if prs else None,
                  "median_margin": float(np.median(m)) if prs else None}
    return res


def stage_effect(model, margin):
    """The learned stage's TSAR-test / NNSeval swaps with and without the margin."""
    from .learned_words_study import MODEL, _rows, all_instances, decide, precision

    stage = pickle.load(open(MODEL, "rb"))
    inst = all_instances()
    rows = _rows()
    out = {}
    for ds in ("tsar_test", "nnseval"):
        rs = [dict(r, f=dict(r["f"])) for r in rows if r["dataset"] == ds]
        dec = decide(stage, rs)
        sw = {k: d for k, d in dec.items() if d["swap"]}
        m = dict(zip(sw, model.margin([d["target"] for d in sw.values()], [d["word"] for d in sw.values()])))
        gated = {k: dict(d, swap=d["swap"] and (k not in m or m[k] >= margin)) for k, d in dec.items()}
        sets = [i for i in inst if i["dataset"] == ds]
        base, new = precision(dec, sets), precision(gated, sets)
        if ds == "tsar_test":
            base["precision_2plus"] = precision(dec, sets, 2)["precision"]
            new["precision_2plus"] = precision(gated, sets, 2)["precision"]
        removed = [(d["target"], d["word"], d["gold"]) for k, d in sw.items() if m[k] < margin]
        out[ds] = {"without": base, "with_margin": new, "removed": len(removed),
                   "removed_gold_correct": sum(g > 0 for *_, g in removed)}
    return out


def fit(argv):
    from cleartext.learned_words import simpler as S

    t0 = time.time()
    data = S.splits()
    model = S.WordDifficulty().fit({s: d["train"] for s, d in data.items()})
    val, vm, vl = _pairs_eval(model, data, "val")
    margin, table = S.choose_margin(vm, vl)
    if margin is None:
        raise SystemExit("REFUSED: no margin reaches the target precision on validation")
    model.margin_threshold = margin
    with open(SIMPLER, "wb") as f:
        pickle.dump(model, f)
    test, tm, tl = _pairs_eval(model, data, "test")
    sel = tm >= margin
    result = {"rule": __doc__.split("Rule (fixed before fitting): ")[1].strip(),
              "train_words": {s: len(d["train"]) for s, d in data.items()},
              "margin": margin, "validation": val, "validation_grid": table,
              "test": test, "test_pairs_at_margin": int(sel.sum()),
              "test_precision_at_margin": float(tl[sel].mean()),
              "public_substitutions": public_substitutions(model, margin),
              "learned_stage_effect": stage_effect(model, margin),
              "seconds": round(time.time() - t0, 1)}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "simpler-check.json").write_text(json.dumps(result, indent=1))
    pre = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "margin": margin,
           "model_sha256": _sha(SIMPLER), "source_sha256": {
               p: _sha(p) for p in ("models/src/cleartext/learned_words/simpler.py",
                                    "evals/learned_words_simpler.py")},
           "note": "written before the round-2 comparison and before any round-3 swap exists"}
    PRECOMMIT.write_text(json.dumps(pre, indent=1))
    print(json.dumps({k: result[k] for k in ("margin", "validation", "test", "test_precision_at_margin",
                                             "public_substitutions", "learned_stage_effect")}, indent=1))
    print("COMPLETE", flush=True)


def round2(argv):
    """Held-out comparison: Claude's marks on the 98 round-2 swaps vs the margin."""
    import csv

    margin, rule = active_margin()
    pre = {"margin": margin, "rule": rule}
    model = pickle.load(open(SIMPLER, "rb"))
    items = {s["key"]: s for s in json.load(open(ROUND2_KEY))["swaps"]}
    marks = {r["key"]: r for r in csv.DictReader(open(ROUND2_MARKS))}
    keys = sorted(k for k in items if k in marks)
    m = model.margin([items[k]["word"] for k in keys], [items[k]["rep"] for k in keys])
    by = Counter()
    for k, x in zip(keys, m):
        by[(marks[k]["mark"], "pass" if x >= pre["margin"] else "block")] += 1
    good = by[("good", "pass")]
    kept = sum(v for (mk, g), v in by.items() if g == "pass")
    out = {"rater": "claude-opus-5-5 (not human), round-2 marks", "swaps": len(keys),
           "margin": margin, "rule": rule, "counts": {f"{a}/{b}": v for (a, b), v in sorted(by.items())},
           "good_rate_before": sum(v for (mk, _), v in by.items() if mk == "good") / len(keys),
           "kept": kept, "good_rate_after": good / kept if kept else None}
    (REPORT / "simpler-round2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    # Per-swap detail (words only) stays local.
    with open(STATE / "rating/round2-simpler.tsv", "w") as f:
        for k, x in sorted(zip(keys, m), key=lambda t: t[1]):
            f.write(f"{x:.2f}\t{marks[k]['mark']}\t{items[k]['word']}\t{items[k]['rep']}\n")


def active_margin():
    """(margin, rule name): R2 when its pre-commitment exists, else the original rule."""
    for path, name in ((PRECOMMIT_R2, "r2"), (PRECOMMIT, "r1")):
        if path.exists():
            pre = json.load(open(path))
            if _sha(SIMPLER) != pre["model_sha256"]:
                raise SystemExit(f"REFUSED: simpler.pkl differs from {path.name}")
            return pre["margin"], name
    raise SystemExit("REFUSED: no simplicity pre-commitment")


def margin_r2(argv):
    from cleartext.learned_words import data as D

    if PRECOMMIT_R2.exists():
        raise SystemExit("REFUSED: R2 already committed")
    model = pickle.load(open(SIMPLER, "rb"))
    pos, neg = [], []
    for i in D.ls_sets():
        if i["dataset"] == "benchls":
            pos += [(i["target"], w) for w in i["gold"]]
    for i in D.tsar("trial"):
        pos += [(i["target"], w) for w in i["gold"]]
    for i in D.swords("dev"):
        neg += [(i["target"], w) for w, share, n in i["rated"] if share >= 0.5 and n >= 3]
    clean = lambda prs: sorted({(a.lower(), b.lower()) for a, b in prs
                                if a.isalpha() and b.isalpha() and a.lower() != b.lower()})
    pos, neg = clean(pos), clean(neg)
    mp = model.margin([a for a, _ in pos], [b for _, b in pos])
    mn = model.margin([a for a, _ in neg], [b for _, b in neg])
    grid = np.round(np.arange(-1.0, 2.001, 0.05), 2)
    table = [{"margin": float(m), "pos_pass": float(np.mean(mp >= m)), "neg_pass": float(np.mean(mn >= m)),
              "balanced_acc": float((np.mean(mp >= m) + np.mean(mn < m)) / 2)} for m in grid]
    best = max(table, key=lambda r: (round(r["balanced_acc"], 6), r["margin"]))
    pre = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rule": "r2 (see learned_words_simpler.py)",
           "margin": best["margin"], "train_side": {"positives": len(pos), "negatives": len(neg), **best},
           "model_sha256": _sha(SIMPLER),
           "source_sha256": {p: _sha(p) for p in ("models/src/cleartext/learned_words/simpler.py",
                                                  "evals/learned_words_simpler.py")},
           "note": "written before the round-2 comparison and before any round-3 swap exists; "
                   "replaces the failed original rule (margin 1.6)"}
    PRECOMMIT_R2.write_text(json.dumps(pre, indent=1))
    out = {"r2": pre, "grid": table, "public_substitutions": public_substitutions(model, best["margin"]),
           "learned_stage_effect": stage_effect(model, best["margin"])}
    (REPORT / "simpler-check-r2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("r2", "public_substitutions", "learned_stage_effect")}, indent=1))
    print("COMPLETE", flush=True)


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "fit":
        return fit(argv[1:])
    if cmd == "margin-r2":
        return margin_r2(argv[1:])
    if cmd == "round2":
        return round2(argv[1:])
    raise SystemExit("usage: learned-words simpler fit|margin-r2|round2")
