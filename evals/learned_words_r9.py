"""Round 9, arm w: sense match and parse stability on top of arm s (see ROUND9.md).

  learned-words r9 senses [--max-seconds N]     SemCor sense table (checkpointed)
  learned-words r9 features [--max-seconds N]   31 arm-s + 13 new features for the 371 dev rows
  learned-words r9 train                        nested CV go/no-go; final fit and pre-commit if go
  learned-words r9 train-h                      round 10: two-head arm h, same gate (ROUND10.md)
  learned-words r9 panel                        fresh passages from the round-6 technical corpus
  learned-words r9 apply [--max-seconds N]      stage + arm s + arm w on the panel (checkpointed)
  learned-words r9 page | score                 three blind rater files; majority-vote scoring
  learned-words r9 apply11 | page11 | score11   round 11: sense-limited candidates (ROUND11.md)

Raw text, features and marks stay in the state directory (r9/).
"""

import csv
import gzip
import hashlib
import json
import pickle
import random
import time
from collections import Counter

import numpy as np

from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r9"
TABLE = DIR / "senses.pkl"
FEATURES = DIR / "features.jsonl"
MODEL = DIR / "model.pkl"
PANEL = DIR / "passages.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r9-precommit.json"
CV_REPORT = REPORT / "r9-cv.json"
MODEL_H = DIR / "model-h.pkl"
PRECOMMIT_H = REPORT / "r10-precommit.json"
CV_REPORT_H = REPORT / "r10-cv.json"
CORPUS = STATE / "corpus"
KEEP = 0.55
REPEATS = 20
SEED = 20261126
RATERS = 3
CONSISTENCY_ITEMS = 20
PER_SOURCE = 45
PASSAGE_SENTENCES = 4


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _threads(argv, default=4):
    import torch

    torch.set_num_threads(int(argv[argv.index("--threads") + 1]) if "--threads" in argv else default)


def _seconds(argv, default):
    return float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else default


def senses(argv):
    from cleartext.learned_words import senses as SN

    _threads(argv)
    DIR.mkdir(exist_ok=True)
    SN.build(TABLE, max_seconds=_seconds(argv, 1500))


def dev_swaps():
    """The 371 development rows: rounds 3-5 (replay items) and the round-8 fresh window.
    Never Joshua's 60, round 2, or the locked test split."""
    items = [dict(s, src=f"replay:{n}") for n, s in enumerate(json.load(open(STATE / "replay/items.json")))]
    mp = json.load(open(STATE / "rating-r8/blind-mixed-map.json"))
    marks = {r["id"]: r["mark"] for r in csv.DictReader(open(STATE / "rating-r8/blind-mixed-marks.csv"))}
    key = {s["key"]: s for s in json.load(open(STATE / "rating-r8/key.json"))["swaps"]}
    r8 = [dict(key[m["key"]], mark=marks[m["id"]], round="r8", src=f"r8:{m['key']}") for m in mp if m["source"] == "r8"]
    return items + r8


def _new_features(sn, s, start):
    from cleartext.learned_words import senses as SN

    return SN.swap_features(sn, s["before"], s["after"], start, s["word"], s["rep"], s["pos"])


def features(argv):
    from cleartext.learned_words import senses as SN

    from .learned_words_simpler import SIMPLER

    _threads(argv)
    deadline = time.monotonic() + _seconds(argv, 1440)
    sn = SN.load(TABLE)
    old = {f"replay:{r['n']}": r["row"] for r in read_jsonl(S.FEATURES)}
    model = simpler = None
    done = {r["src"] for r in read_jsonl(FEATURES)}
    with open(FEATURES, "a") as out:
        for s in dev_swaps():
            if s["src"] in done:
                continue
            if time.monotonic() > deadline:
                break
            row = old.get(s["src"])
            if s["src"].startswith("r8:"):
                if model is None:
                    model = pickle.load(open(STATE / "model.pkl", "rb"))
                    simpler = pickle.load(open(SIMPLER, "rb"))
                row = S.swap_row(model, simpler, s)
            if row is not None:
                row = dict(row, new=_new_features(sn, s, row["start"]))
            out.write(json.dumps({"src": s["src"], "row": row}) + "\n")
            out.flush()
    rows = read_jsonl(FEATURES)
    left = len(dev_swaps()) - len({r["src"] for r in rows})
    print(f"{len(rows)} rows, {sum(r['row'] is None for r in rows)} unresolved, {left} left", flush=True)
    print("COMPLETE" if not left else "PARTIAL", flush=True)


def matrices(T, ids, swaps, rows):
    """(31 arm-s features, 44 arm-w features) for resolved swaps."""
    from cleartext.learned_words import senses as SN

    A = np.array([[r["f"][k] for k in S.STAGE_KEYS] for r in rows], float)
    spans = [dict(s, start=r["start"]) for s, r in zip(swaps, rows)]
    Xs = np.nan_to_num(np.hstack([A, S.rewrite_features(T, ids, spans)]))
    N = np.nan_to_num(np.array([[r["new"][k] for k in SN.KEYS] for r in rows], float))
    return Xs, np.hstack([Xs, N])


def dataset():
    from .learned_words_rewrite import load

    _, T, ids, _ = load()
    by_src = {s["src"]: s for s in dev_swaps()}
    rows = [r for r in read_jsonl(FEATURES) if r["row"] is not None]
    swaps = [by_src[r["src"]] for r in rows]
    Xs, Xw = matrices(T, ids, swaps, [r["row"] for r in rows])
    marks = np.array([s["mark"] for s in swaps])
    b = np.array(["b" in s["arms"] for s in swaps])
    groups = np.array([r["row"]["lemma"] for r in rows])
    return Xs, Xw, marks, b, groups, swaps


def nested(X, y, b, groups, seed):
    from sklearn.model_selection import GroupKFold

    score, keep = np.zeros(len(y)), np.zeros(len(y), bool)
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        inner = S._oof(X[tr], y[tr], groups[tr], seed + 1)
        t = np.quantile(inner[b[tr]], 1 - KEEP)
        score[te] = S._fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        keep[te] = score[te] >= t
    return score, keep


def train(argv):
    from sklearn.metrics import roc_auc_score

    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    Xs, Xw, marks, b, groups, swaps = dataset()
    if len(marks) != len(dev_swaps()) or b.sum() != 189:
        raise SystemExit(f"REFUSED: {len(marks)} of {len(dev_swaps())} rows, {b.sum()} of 189 arm-b rows resolved")
    y = (marks == "good").astype(int)
    base = float(y[b].mean())
    meaning_b = int((marks[b] == "meaning").sum())
    res = {"rows": len(y), "arm_b_rows": int(b.sum()), "good_rate_base": base, "meaning_base": meaning_b,
           "keep": KEEP, "repeats": REPEATS}
    for arm, X in (("s", Xs), ("w", Xw)):
        runs = []
        for k in range(REPEATS):
            score, keep = nested(X, y, b, groups, SEED + 10 * k)
            kb = keep & b
            runs.append({"good_rate": float(y[kb].mean()) if kb.any() else 0.0,
                         "keep_share": float(kb.sum() / b.sum()),
                         "meaning_removed": float(1 - (marks[kb] == "meaning").sum() / meaning_b),
                         "errors_removed": float(1 - (marks[kb] != "good").sum() / (marks[b] != "good").sum()),
                         "auc_good_b": float(roc_auc_score(y[b], score[b])),
                         "auc_meaning_b": float(roc_auc_score(marks[b] == "meaning", -score[b]))})
        res[arm] = {"mean": {k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
                    "min_max": {k: [float(np.min([r[k] for r in runs])), float(np.max([r[k] for r in runs]))]
                                for k in runs[0]},
                    "runs_good_rate": [r["good_rate"] for r in runs]}
    w, s = res["w"]["mean"], res["s"]["mean"]
    paired = float(np.mean(np.array(res["w"]["runs_good_rate"]) - np.array(res["s"]["runs_good_rate"])))
    res["paired_good_gain"] = paired
    res["go"] = bool(paired >= 0.03 and w["meaning_removed"] >= 0.5 and w["good_rate"] >= base + 0.10)
    res["features_sha256"] = _sha(FEATURES)
    res["table_sha256"] = _sha(TABLE)
    res["source_sha256"] = {p: _sha(p) for p in ("evals/learned_words_r9.py", "models/src/cleartext/learned_words/senses.py")}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("s", "w")} | {"s": s, "w": w}, indent=1))
    if res["go"]:
        oof = S._oof(Xw, y, groups, SEED)
        t = float(np.quantile(oof[b], 1 - KEEP))
        pickle.dump(S._fit(Xw, y), open(MODEL, "wb"))
        json.dump({"threshold": t, "keep": KEEP, "model_sha256": _sha(MODEL), "table_sha256": _sha(TABLE),
                   "features_sha256": _sha(FEATURES), "arm_s_precommit_sha256": _sha(S.PRECOMMIT),
                   "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
        print(f"pre-committed threshold {t:.4f}")
    print("COMPLETE")


# --- round 10: two heads (ROUND10.md) -----------------------------------------------

class TwoHeads:
    """P(Good) = (1 - P(Changes meaning)) * (1 - P(Not simpler)), each head a logistic on 44 features."""

    def __init__(self, X, marks):
        self.m = S._fit(X, (marks == "meaning").astype(int))
        self.n = S._fit(X, (marks == "notsimpler").astype(int))

    def predict_proba(self, X):
        p = (1 - self.m.predict_proba(X)[:, 1]) * (1 - self.n.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def _oof_h(X, marks, groups, seed):
    from sklearn.model_selection import GroupKFold

    p = np.zeros(len(marks))
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, marks, groups):
        p[te] = TwoHeads(X[tr], marks[tr]).predict_proba(X[te])[:, 1]
    return p


def nested_h(X, marks, b, groups, seed):
    from sklearn.model_selection import GroupKFold

    score, keep = np.zeros(len(marks)), np.zeros(len(marks), bool)
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, marks, groups):
        inner = _oof_h(X[tr], marks[tr], groups[tr], seed + 1)
        t = np.quantile(inner[b[tr]], 1 - KEEP)
        score[te] = TwoHeads(X[tr], marks[tr]).predict_proba(X[te])[:, 1]
        keep[te] = score[te] >= t
    return score, keep


def _runs(nest, y, marks, b, groups, seed_base):
    from sklearn.metrics import roc_auc_score

    meaning_b = (marks[b] == "meaning").sum()
    runs = []
    for k in range(REPEATS):
        score, keep = nest(seed_base + 10 * k)
        kb = keep & b
        runs.append({"good_rate": float(y[kb].mean()) if kb.any() else 0.0,
                     "keep_share": float(kb.sum() / b.sum()),
                     "meaning_removed": float(1 - (marks[kb] == "meaning").sum() / meaning_b),
                     "errors_removed": float(1 - (marks[kb] != "good").sum() / (marks[b] != "good").sum()),
                     "auc_good_b": float(roc_auc_score(y[b], score[b])),
                     "auc_meaning_b": float(roc_auc_score(marks[b] == "meaning", -score[b]))})
    return {"mean": {k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
            "min_max": {k: [float(np.min([r[k] for r in runs])), float(np.max([r[k] for r in runs]))] for k in runs[0]},
            "runs_good_rate": [r["good_rate"] for r in runs]}


def train_h(argv):
    if PRECOMMIT_H.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT_H.name} exists")
    if json.load(open(CV_REPORT))["features_sha256"] != _sha(FEATURES):
        raise SystemExit("REFUSED: features differ from round 9")
    Xs, Xw, marks, b, groups, swaps = dataset()
    y = (marks == "good").astype(int)
    base = float(y[b].mean())
    res = {"rows": len(y), "arm_b_rows": int(b.sum()), "good_rate_base": base,
           "meaning_base": int((marks[b] == "meaning").sum()), "keep": KEEP, "repeats": REPEATS}
    res["s"] = _runs(lambda sd: nested(Xs, y, b, groups, sd), y, marks, b, groups, SEED)
    res["h"] = _runs(lambda sd: nested_h(Xw, marks, b, groups, sd), y, marks, b, groups, SEED)
    h = res["h"]["mean"]
    res["paired_good_gain"] = float(np.mean(np.array(res["h"]["runs_good_rate"]) - np.array(res["s"]["runs_good_rate"])))
    res["go"] = bool(res["paired_good_gain"] >= 0.03 and h["meaning_removed"] >= 0.5 and h["good_rate"] >= base + 0.10)
    res["features_sha256"] = _sha(FEATURES)
    res["table_sha256"] = _sha(TABLE)
    res["source_sha256"] = {p: _sha(p) for p in ("evals/learned_words_r9.py", "models/src/cleartext/learned_words/senses.py")}
    json.dump(res, open(CV_REPORT_H, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("s", "h")} | {"s": res["s"]["mean"], "h": h}, indent=1))
    if res["go"]:
        oof = _oof_h(Xw, marks, groups, SEED)
        t = float(np.quantile(oof[b], 1 - KEEP))
        pickle.dump(TwoHeads(Xw, marks), open(MODEL_H, "wb"))
        json.dump({"threshold": t, "keep": KEEP, "model_sha256": _sha(MODEL_H), "table_sha256": _sha(TABLE),
                   "features_sha256": _sha(FEATURES), "arm_s_precommit_sha256": _sha(S.PRECOMMIT),
                   "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT_H, "w"), indent=1)
        print(f"pre-committed threshold {t:.4f}")
    print("COMPLETE")


def load_h():
    pre = json.load(open(PRECOMMIT_H))
    if _sha(MODEL_H) != pre["model_sha256"] or _sha(TABLE) != pre["table_sha256"]:
        raise SystemExit("REFUSED: arm h model or sense table differ from the pre-commitment")
    return pickle.load(open(MODEL_H, "rb")), pre["threshold"]


# --- fresh window ------------------------------------------------------------------

def _ok_sentence(s):
    words = s.split()
    letters = sum(c.isalpha() for c in s)
    return 8 <= len(words) <= 50 and letters >= 0.7 * len(s.replace(" ", "")) and s[-1:] in ".;:)"


def panel(argv):
    """PER_SOURCE passages of PASSAGE_SENTENCES consecutive clean sentences, one per document,
    from a seeded shuffle of each corpus source's documents."""
    from cleartext.learned_words.rewrite import tokens

    if PANEL.exists():
        raise SystemExit(f"REFUSED: {PANEL} exists")
    used = []
    for p in ("panel-r4", "panel-r5", "panel-r8"):
        used += [x["original"] for x in json.load(open(STATE / p / "passages.json"))]
    used += [s["before"] for s in dev_swaps()]
    grams = set()
    for t in used:
        tt = tokens(t)
        grams |= {tuple(tt[i:i + 8]) for i in range(len(tt) - 7)}
    out = []
    for src in ("rfc", "fedreg", "openfda"):
        docs = {}
        for line in gzip.open(CORPUS / f"{src}.jsonl.gz", "rt"):
            r = json.loads(line)
            docs.setdefault(r["doc"], []).append(r["s"])
        order = sorted(docs)
        rng = random.Random(f"{SEED}:{src}")
        rng.shuffle(order)
        n = 0
        for d in order:
            if n >= PER_SOURCE:
                break
            sents = docs[d]
            starts = [i for i in range(len(sents) - PASSAGE_SENTENCES + 1)
                      if all(_ok_sentence(x) for x in sents[i:i + PASSAGE_SENTENCES])]
            if not starts:
                continue
            i = rng.choice(starts)
            text = " ".join(sents[i:i + PASSAGE_SENTENCES])
            tt = tokens(text)
            if any(tuple(tt[k:k + 8]) in grams for k in range(len(tt) - 7)):
                continue
            n += 1
            out.append({"id": f"r9-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": d,
                        "first_sentence": i, "original": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    json.dump(out, open(PANEL, "w"), indent=1)
    print(json.dumps({"passages": len(out), "by_source": Counter(p["domain"] for p in out)}, indent=1))
    print("COMPLETE")


def load_w():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL) != pre["model_sha256"] or _sha(TABLE) != pre["table_sha256"]:
        raise SystemExit("REFUSED: arm w model or sense table differ from the pre-commitment")
    return pickle.load(open(MODEL, "rb")), pre["threshold"]


def apply(argv):
    from cleartext.learned_words import senses as SN

    from .learned_words_apply import word_swaps
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    _threads(argv)
    deadline = time.monotonic() + _seconds(argv, 1380)
    clf_s, _ = S.load_filter()
    clf_w, _ = load_h()  # round 10: arm h (arm w did not pass round 9)
    sn = SN.load(TABLE)
    _, T, ids, _ = load()
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    simpler = pickle.load(open(SIMPLER, "rb"))
    pan = json.load(open(PANEL))
    skip = _rewrite_overlap(pan)
    done = {r["pid"] for r in read_jsonl(APPLY)}
    todo = [p for p in pan if p["id"] not in done and p["id"] not in skip]
    print(f"{len(done)} done, {len(todo)} to do, {len(skip)} skipped for rewrite-corpus overlap", flush=True)
    with open(APPLY, "a") as f:
        for p in todo:
            if time.monotonic() > deadline:
                break
            swaps, seen = word_swaps(p["original"], model, p["id"])
            got, rows = [], []
            for s in swaps:
                r = S.swap_row(model, simpler, s)
                if r is not None:
                    got.append(s)
                    rows.append(dict(r, new=_new_features(sn, s, r["start"])))
            if got:
                Xs, Xw = matrices(T, ids, got, rows)
                for s, r, a, w in zip(got, rows, clf_s.predict_proba(Xs)[:, 1], clf_w.predict_proba(Xw)[:, 1]):
                    s.update(margin=r["f"]["margin"], stack=float(a), w=float(w), new=r["new"])
            out = [s for s in swaps if "w" in s]
            f.write(json.dumps({"pid": p["id"], "domain": p["domain"], "kind": p["kind"], "content_words": seen,
                                "unresolved": len(swaps) - len(out), "swaps": out}) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} swaps={len(out)}", flush=True)
    left = len({p["id"] for p in pan} - skip - {r["pid"] for r in read_jsonl(APPLY)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


def _rewrite_overlap(pan):
    """Passage ids sharing a word 8-gram with the rewrite corpora (cached)."""
    from cleartext.learned_words.rewrite import tokens

    from .learned_words_rewrite import pairs

    path = DIR / "panel-overlap.json"
    if path.exists():
        return set(json.load(open(path)))
    index = {}
    for p in pan:
        t = tokens(p["original"])
        for i in range(len(t) - 7):
            index.setdefault(tuple(t[i:i + 8]), set()).add(p["id"])
    hit = set()
    for _, _, a, b in pairs():
        for x in (a, b):
            t = tokens(x)
            for i in range(len(t) - 7):
                hit |= index.get(tuple(t[i:i + 8]), set())
    json.dump(sorted(hit), open(path, "w"))
    return hit


def picks(swaps, ts, tw, choose):
    """(arm b, arm s, arm w) swaps of one passage. Filters apply first, then the per-passage limits."""
    ok = [s for s in swaps if s["margin"] >= S.MIN_MARGIN]
    return choose(ok), choose([s for s in ok if s["stack"] >= ts]), choose([s for s in ok if s["w"] >= tw])


def page(argv):
    from .learned_words_apply import _skey, choose

    _, ts = S.load_filter()
    _, tw = load_h()
    items = {}
    for r in read_jsonl(APPLY):
        for arm, chosen in zip("bsw", picks(r["swaps"], ts, tw, choose)):
            for s in chosen:
                k = _skey(s)
                items.setdefault(k, dict(s, key=k, arms=[], domain=r["domain"]))["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1126)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r9") for x in items] + [dict(x, source="old", old_mark=x["mark"]) for x in old[:CONSISTENCY_ITEMS]]
    RATING.mkdir(exist_ok=True)
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r9" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    for r in range(1, RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1126:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r9_items": len(items), "items": len(base),
                      "per_arm": Counter(a for x in items for a in x["arms"])}, indent=1))


SEVERITY = ("meaning", "notsimpler", "good")


def majority(ms):
    c = Counter(ms)
    top, n = c.most_common(1)[0]
    if n >= 2:
        return top
    return min(ms, key=SEVERITY.index)


def score(argv):
    from .learned_words_study import wilson

    mp = json.load(open(RATING / "blind-map.json"))
    ids = {m["id"] for m in mp}
    raters = []
    for r in range(1, RATERS + 1):
        marks = {x["id"]: x["mark"] for x in csv.DictReader(open(RATING / f"marks-{r}.csv"))}
        if set(marks) != ids or set(marks.values()) - set(SEVERITY):
            raise SystemExit(f"REFUSED: rater {r} marks are incomplete or use an unknown label")
        raters.append(marks)
    maj = {i: majority([m[i] for m in raters]) for i in ids}
    out = {"items": len(ids), "raters": RATERS}

    def arm_stats(marks):
        res = {}
        for arm in "bsw":
            ms = [marks[m["id"]] for m in mp if m["source"] == "r9" and arm in m["arms"]]
            c = Counter(ms)
            res["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(c["good"], len(ms)), "meaning_rate": c["meaning"] / max(1, len(ms))}
        return res

    out["majority"] = arm_stats(maj)
    out["per_rater"] = [arm_stats(m) for m in raters]
    b, w = out["majority"]["arm_b"], out["majority"]["arm_w"]
    out["keep_share_w"] = w["swaps"] / max(1, b["swaps"])
    out["keep_share_s"] = out["majority"]["arm_s"]["swaps"] / max(1, b["swaps"])
    out["success"] = bool(w["good_rate"] >= 0.78 and w["meaning_rate"] <= 0.15 and out["keep_share_w"] >= 0.45)
    r9 = [m["id"] for m in mp if m["source"] == "r9"]
    out["unanimous_share"] = sum(len({m[i] for m in raters}) == 1 for i in r9) / max(1, len(r9))
    pairs = [(a, c) for a in range(RATERS) for c in range(a + 1, RATERS)]
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r9) / max(1, len(r9))
                                 for a, c in pairs}
    old = [m for m in mp if m["source"] == "old"]
    out["consistency_with_old"] = [sum(m["old_mark"] == rt[m["id"]] for m in old) for rt in raters] + [len(old)]
    (REPORT / "round10.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# --- round 11: sense-limited candidates (ROUND11.md) --------------------------------

SENSE_COVER = 0.25
APPLY11 = DIR / "apply11.jsonl"
RATING11 = DIR / "rating11"


def in_sense(sn, text, start, end, lemma, pos, words):
    """For each candidate word: the target's in-context WSD mass on senses the candidate shares."""
    from nltk.corpus import wordnet as wn

    from cleartext.learned_words import features as F
    from cleartext.learned_words import senses as SN

    ctx = SN.contextual([(text, start, end)])[0]
    pt, _ = sn.wsd(ctx, lemma, pos)
    allowed = ("a", "s") if pos == "a" else (pos,)
    out = []
    for w in words:
        c = F.lemma_of(w, F.UPOS[pos]).replace(" ", "_")
        syns = {x for x in wn.synsets(c) if x.pos() in allowed}
        out.append(float(sum(p for x, p in pt.items() if x in syns)))
    return out


def _decide_top(model, F, decide, inst, rows, s):
    scores = model["stage"].rank(rows)
    top = max(zip(scores, rows), key=lambda x: x[0])[1]
    a, b = F.window(s, inst["start"], inst["end"], top["word"])
    fw, bw = F.entail([(a, b), (b, a)])
    top["f"].update({"nli_fw": fw, "nli_bw": bw, "nli_min": min(fw, bw)})
    d = decide(model, rows)[inst["id"]]
    return d if d["word"] == top["word"] else None, top


def word_swaps_two(text, model, pid, sn):
    """The stage's swaps for one passage twice from one featurization: all candidates (arm b's
    generator) and only candidates covering at least SENSE_COVER of the target's in-context sense
    mass (arm g's generator). Mirrors learned_words_apply.word_swaps."""
    from cleartext.features import nlp
    from cleartext.learned_words import features as F

    from .learned_words_apply import MAX_TARGETS, sent, targets
    from .learned_words_study import decide

    pool, seen = [], 0
    for si, s in enumerate(sent(text)):
        found = targets(nlp()(s), model, limit=1000)
        if found:
            elig, allc = found
            seen += len(allc)
            pool += [(pc * (1 - pt), si, s, t, pc, pt) for t, pc, pt in elig]
    out = {"all": [], "sense": []}
    for _, si, s, t, pc, pt in sorted(pool, key=lambda x: -x[0])[:MAX_TARGETS]:
        iid = f"{pid}:{si}:{t.idx}"
        inst = {"id": iid, "dataset": "apply", "group": t.lemma_.lower(), "text": s, "start": t.idx,
                "end": t.idx + len(t), "target": t.text, "gold": {}, "rated": []}
        rows = F.featurize(inst, given=False)
        if not rows:
            continue
        model["extra"][iid] = {"p_complex": pc, "p_term": pt}
        for r in rows:
            r["f"].update(model["extra"][iid])
        cover = in_sense(sn, s, inst["start"], inst["end"], rows[0]["lemma"], rows[0]["pos"], [r["word"] for r in rows])
        variants = {"all": rows, "sense": [r for r, c in zip(rows, cover) if c >= SENSE_COVER]}
        for name, rs in variants.items():
            if not rs:
                continue
            d, top = _decide_top(model, F, decide, inst, rs, s)
            if d is None or not d["swap"]:
                continue
            after = s[:inst["start"]] + d["word"] + s[inst["end"]:]
            out[name].append({"pid": pid, "sentence": si, "before": s, "after": after, "word": t.text,
                              "rep": d["word"], "pos": d["pos"], "gate": d["gate"], "check": d["check"],
                              "p_complex": pc, "p_term": pt, "sources": top["sources"],
                              "cover": cover[rows.index(top)] if top in rows else None})
    return out, seen


def apply11(argv):
    from cleartext.learned_words import senses as SN

    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    _threads(argv)
    pre = json.load(open(REPORT / "r11-precommit.json"))
    if _sha(TABLE) != pre["table_sha256"] or _sha(S.PRECOMMIT) != pre["arm_s_precommit_sha256"]:
        raise SystemExit("REFUSED: sense table or arm s differ from the round-11 pre-commitment")
    deadline = time.monotonic() + _seconds(argv, 1380)
    clf_s, _ = S.load_filter()
    sn = SN.load(TABLE)
    _, T, ids, _ = load()
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    simpler = pickle.load(open(SIMPLER, "rb"))
    pan = json.load(open(PANEL))
    skip = _rewrite_overlap(pan)
    done = {r["pid"] for r in read_jsonl(APPLY11)}
    todo = [p for p in pan if p["id"] not in done and p["id"] not in skip]
    print(f"{len(done)} done, {len(todo)} to do, {len(skip)} skipped for rewrite-corpus overlap", flush=True)
    with open(APPLY11, "a") as f:
        for p in todo:
            if time.monotonic() > deadline:
                break
            variants, seen = word_swaps_two(p["original"], model, p["id"], sn)
            rec = {"pid": p["id"], "domain": p["domain"], "content_words": seen}
            for name, swaps in variants.items():
                got, rows = [], []
                for s in swaps:
                    r = S.swap_row(model, simpler, s)
                    if r is not None:
                        got.append(s)
                        rows.append(r)
                if got:
                    for s, r, x in zip(got, rows, S.score_swaps(clf_s, T, ids, got, rows)):
                        s.update(margin=r["f"]["margin"], stack=float(x))
                rec[name] = [s for s in swaps if "stack" in s]
                rec[f"unresolved_{name}"] = len(swaps) - len(rec[name])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} all={len(rec['all'])} sense={len(rec['sense'])}", flush=True)
    left = len({p["id"] for p in pan} - skip - {r["pid"] for r in read_jsonl(APPLY11)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


ARMS11 = ("b", "s", "g", "gs")


def picks11(rec, ts, choose):
    """Arm b: all candidates, margin. Arm s: b + arm-s filter. Arm g: sense-limited candidates, margin.
    Arm gs: g + arm-s filter. Filters apply before the per-passage limits."""
    a = [s for s in rec["all"] if s["margin"] >= S.MIN_MARGIN]
    g = [s for s in rec["sense"] if s["margin"] >= S.MIN_MARGIN]
    return {"b": choose(a), "s": choose([s for s in a if s["stack"] >= ts]),
            "g": choose(g), "gs": choose([s for s in g if s["stack"] >= ts])}


def page11(argv):
    from .learned_words_apply import _skey, choose

    _, ts = S.load_filter()
    items = {}
    for r in read_jsonl(APPLY11):
        for arm, chosen in picks11(r, ts, choose).items():
            for s in chosen:
                k = _skey(s)
                items.setdefault(k, dict(s, key=k, arms=[], domain=r["domain"]))["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1127)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r11") for x in items] + [dict(x, source="old", old_mark=x["mark"]) for x in old[:CONSISTENCY_ITEMS]]
    RATING11.mkdir(exist_ok=True)
    json.dump({"swaps": items}, open(RATING11 / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r11" else None} for n, x in enumerate(base, 1)],
              open(RATING11 / "blind-map.json", "w"), indent=1)
    for r in range(1, RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1127:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING11 / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r11_items": len(items), "items": len(base),
                      "per_arm": Counter(a for x in items for a in x["arms"])}, indent=1))


def score11(argv):
    from .learned_words_study import wilson

    mp = json.load(open(RATING11 / "blind-map.json"))
    ids = {m["id"] for m in mp}
    raters = []
    for r in range(1, RATERS + 1):
        marks = {x["id"]: x["mark"] for x in csv.DictReader(open(RATING11 / f"marks-{r}.csv"))}
        if set(marks) != ids or set(marks.values()) - set(SEVERITY):
            raise SystemExit(f"REFUSED: rater {r} marks are incomplete or use an unknown label")
        raters.append(marks)
    maj = {i: majority([m[i] for m in raters]) for i in ids}

    def arm_stats(marks):
        res = {}
        for arm in ARMS11:
            ms = [marks[m["id"]] for m in mp if m["source"] == "r11" and arm in m["arms"]]
            c = Counter(ms)
            res["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(c["good"], len(ms)), "meaning_rate": c["meaning"] / max(1, len(ms))}
        return res

    out = {"items": len(ids), "raters": RATERS, "majority": arm_stats(maj), "per_rater": [arm_stats(m) for m in raters]}
    b, gs = out["majority"]["arm_b"], out["majority"]["arm_gs"]
    out["yield_gs"] = gs["swaps"] / max(1, b["swaps"])
    out["success"] = bool(gs["good_rate"] >= 0.78 and gs["meaning_rate"] <= 0.15 and out["yield_gs"] >= 0.40)
    r11 = [m["id"] for m in mp if m["source"] == "r11"]
    out["unanimous_share"] = sum(len({m[i] for m in raters}) == 1 for i in r11) / max(1, len(r11))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r11) / max(1, len(r11))
                                 for a in range(RATERS) for c in range(a + 1, RATERS)}
    old = [m for m in mp if m["source"] == "old"]
    out["consistency_with_old"] = [sum(m["old_mark"] == rt[m["id"]] for m in old) for rt in raters] + [len(old)]
    (REPORT / "round11.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"senses": senses, "features": features, "train": train, "train-h": train_h, "panel": panel, "apply": apply,
           "page": page, "score": score, "apply11": apply11, "page11": page11, "score11": score11}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r9 senses|features|train|train-h|panel|apply|page|score|apply11|page11|score11")
