"""Round 14: arm m, arm p plus two meaning signals, refitted with round 12's marks (see ROUND14.md).

  learned-words r14 train                    CV gate; final fit and pre-commit if go
  learned-words r14 apply [--max-seconds N]  fresh panel (once) and round 11's two generators (checkpointed)
  learned-words r14 page | score             three blind rater files; majority-vote scoring

Raw text, pools and marks stay in the state directory (r14/).
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

from . import learned_words_r12 as P12
from . import learned_words_r13 as P13
from . import learned_words_r9 as R
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r14"
POOL = DIR / "pool.json"
MODEL = DIR / "model.pkl"
PANEL = DIR / "passages.json"
OVERLAP = DIR / "panel-overlap.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r14-precommit.json"
CV_REPORT = REPORT / "r14-cv.json"
RESULT = REPORT / "round14.json"
SEED = 20261214
GRID = (0.40, 0.45, 0.50, 0.55, 0.60)  # candidate keeps for arm m
TARGET_R12 = 0.92  # keep rule: the widest keep whose CV Good rate on round 12's arm-b rows reaches this
WIDER = 0.10  # arm mw: the same model at a keep 10 points wider (secondary)
REPEATS = 10
PER_SOURCE = 30
EXTRA = ("ctx_cos", "check")  # two meaning signals from the 31 arm-s features
ARMS = ("b", "s", "gs", "p", "m", "mw")
SOURCES = ("evals/learned_words_r14.py", "evals/learned_words_r13.py") + P12.SOURCES


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _rated(rating, apply_path, rnd):
    """A round's rated swaps with majority marks, arm-s score, cover and 31 arm-s features."""
    from .learned_words_apply import _skey

    mp = json.load(open(rating / "blind-map.json"))
    raters = [{x["id"]: x["mark"] for x in csv.DictReader(open(rating / f"marks-{r}.csv"))}
              for r in range(1, P12.RATERS + 1)]
    x31 = {}
    for rec in read_jsonl(apply_path):
        for name in ("all", "next", "sense"):
            for s in rec.get(name, []):
                x31[_skey(s)] = s["x31"]
    key = {s["key"]: s for s in json.load(open(rating / "key.json"))["swaps"]}
    out = []
    for m in mp:
        if m["source"] != rnd:
            continue
        s = key[m["key"]]
        out.append({"src": f"{rnd}:{m['key']}", "round": rnd, "group": f"{rnd}:{s['pid']}", "word": s["word"],
                    "rep": s["rep"], "pos": s["pos"], "mark": R.majority([rt[m["id"]] for rt in raters]),
                    "b": "b" in m["arms"], "stack": float(s["stack"]), "cover": float(s["cover"] or 0.0),
                    "x31": x31[m["key"]], "runner_up": bool(s.get("runner_up"))})
    return out


def pool(with_r13=False):
    """Round 12's frozen 506-row pool (rounds 3-8 and 11) with the 31 arm-s features attached,
    plus rounds 12 and 13's rated swaps."""
    rows = json.load(open(P12.POOL))
    Xs, _, _, _, _, swaps = R.dataset()
    dev_x = {s["src"]: [float(v) for v in x] for s, x in zip(swaps, Xs)}
    f11 = {r["key"]: r for r in read_jsonl(P13.FEATURES11)}
    for r in rows:
        r["x31"] = f11[r["src"].split(":", 1)[1]]["x31"] if r["round"] == "r11" else dev_x[r["src"]]
        r["runner_up"] = False
    rows += _rated(P12.RATING, P12.APPLY, "r12")
    if with_r13:
        rows += _rated(P13.RATING, P13.APPLY, "r13")
    for r in rows:
        r["key"] = list(P12.pair_key(r["word"], r["rep"], r["pos"]))
        r["good"] = int(r["mark"] == "good")
    return rows




class ArmM(P12.ArmP):
    """Arm p's four features plus ctx_cos and the stage checker score; same logistic (C = 0.1)."""

    IDX = [S.KEYS.index(k) for k in EXTRA]

    def _design(self, rows, loo):
        X = P12.design(rows, self.pri, loo)
        E = np.array([[r["x31"][i] for i in self.IDX] for r in rows], float)
        return np.hstack([X, E])

    def __init__(self, rows):
        self.pri = P12.Priors([r["key"] for r in rows], [r["good"] for r in rows])
        self.clf = S._fit(self._design(rows, True), np.array([r["good"] for r in rows]))

    def score(self, rows):
        return self.clf.predict_proba(self._design(rows, False))[:, 1]


def oof(cls, rows, groups, seed):
    p = np.zeros(len(rows))
    for tr, te in P12._folds(groups, seed):
        p[te] = cls([rows[i] for i in tr]).score([rows[i] for i in te])
    return p


def nested(cls, rows, groups, ref, seed, keep):
    """Outer out-of-fold scores and keep decisions; thresholds from inner out-of-fold scores on the
    training folds' reference rows."""
    score, kp = np.zeros(len(rows)), np.zeros(len(rows), bool)
    for tr, te in P12._folds(groups, seed):
        sub = [rows[i] for i in tr]
        inner = oof(cls, sub, groups[tr], seed + 1)
        t = np.quantile(inner[ref[tr]], 1 - keep)
        score[te] = cls(sub).score([rows[i] for i in te])
        kp[te] = score[te] >= t
    return score, kp


def train(argv):
    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    DIR.mkdir(exist_ok=True)
    rows = pool()
    marks = np.array([r["mark"] for r in rows])
    y = np.array([r["good"] for r in rows])
    b = np.array([r["b"] for r in rows])
    rnd = np.array([r["round"] for r in rows])
    ref = b & np.isin(rnd, ["r11", "r12"])
    ref12 = ref & (rnd == "r12")
    if len(rows) != 620 or ref.sum() != 233:
        raise SystemExit(f"REFUSED: pool has {len(rows)} rows, {ref.sum()} round-11/12 arm-b rows")
    json.dump(rows, open(POOL, "w"))
    groups = np.array([r["group"] for r in rows])
    res = {"rows": len(rows), "ref_rows": int(ref.sum()), "ref_r12_rows": int(ref12.sum()), "grid": list(GRID),
           "target_r12": TARGET_R12, "repeats": REPEATS, "extra": list(EXTRA)}

    # A. passage-grouped nested CV over the keep grid: arm m vs arm p refitted on the same pool
    grid = {}
    for keep in GRID:
        runs = []
        for k in range(REPEATS):
            sm, km = nested(ArmM, rows, groups, ref, P12.SEED + 10 * k, keep)
            sp, kp = nested(P12.ArmP, rows, groups, ref, P12.SEED + 10 * k, keep)
            runs.append({"m": P12._stats(km, marks, ref), "p": P12._stats(kp, marks, ref),
                         "m_r12": P12._stats(km, marks, ref12), "p_r12": P12._stats(kp, marks, ref12),
                         "auc_m": P12._auc(y[ref], sm[ref]), "auc_p": P12._auc(y[ref], sp[ref])})
        g = {a: {f: float(np.mean([r[a][f] for r in runs])) for f in ("kept", "keep_share", "good_rate", "meaning")}
             for a in ("m", "p", "m_r12", "p_r12")}
        g["paired_gain_m_vs_p"] = float(np.mean([r["m"]["good_rate"] - r["p"]["good_rate"] for r in runs]))
        g["auc_m"] = float(np.mean([r["auc_m"] for r in runs]))
        g["auc_p"] = float(np.mean([r["auc_p"] for r in runs]))
        grid[str(keep)] = g
    res["passage_cv"] = grid
    ok = [k for k in GRID if grid[str(k)]["m_r12"]["good_rate"] >= TARGET_R12]
    keep = max(ok) if ok else min(GRID)
    res["keep"] = keep
    pc = grid[str(keep)]

    # B. time split: fit on rounds 3-11 (round 12's pool), thresholds from round 11's arm-b rows, score round 12
    early = rnd != "r12"
    e_rows = [r for r, x in zip(rows, early) if x]
    r12 = rnd == "r12"
    l_rows = [r for r, x in zip(rows, r12) if x]
    ts = {}
    for name, cls in (("m", ArmM), ("p", P12.ArmP)):
        inner = np.mean([oof(cls, e_rows, groups[early], P12.SEED + k) for k in range(5)], axis=0)
        t = np.quantile(inner[ref[early]], 1 - keep)
        sc = np.zeros(len(rows))
        sc[r12] = cls(e_rows).score(l_rows)
        ts[name] = dict(P12._stats(r12 & (sc >= t), marks, ref12), auc=P12._auc(y[ref12], sc[ref12]))
    res["time_split_r12"] = ts

    # C. diagnostic: lemma-grouped (no priors for held-out rows)
    lem = np.array([r["key"][0] for r in rows])
    lr = [(nested(ArmM, rows, lem, ref, P12.SEED + 10 * k, keep)[1], nested(P12.ArmP, rows, lem, ref, P12.SEED + 10 * k, keep)[1])
          for k in range(5)]
    res["lemma_cv_diagnostic"] = {a: float(np.mean([P12._stats(x[j], marks, ref)["good_rate"] for x in lr]))
                                  for j, a in enumerate(("m", "p"))}
    res["go_conditions"] = {"keep_rule_met": bool(ok),
                            "cv_good_ge_0.90": pc["m"]["good_rate"] >= 0.90,
                            "cv_gain_vs_p_ge_0.02": pc["paired_gain_m_vs_p"] >= 0.02,
                            "time_split_gain_ge_0.03_keep_ge_0.35": ts["m"]["good_rate"] >= ts["p"]["good_rate"] + 0.03
                            and ts["m"]["keep_share"] >= 0.35}
    res["go"] = bool(all(res["go_conditions"].values()))
    res["pool_sha256"] = _sha(POOL)
    res["source_sha256"] = {p: _sha(p) for p in SOURCES}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "source_sha256"}, indent=1))
    if res["go"]:
        full = np.mean([oof(ArmM, rows, groups, P12.SEED + k) for k in range(REPEATS)], axis=0)
        thr = {"m": float(np.quantile(full[ref], 1 - keep)), "mw": float(np.quantile(full[ref], 1 - keep - WIDER))}
        pickle.dump(ArmM(rows), open(MODEL, "wb"))
        pre12 = json.load(open(P12.PRECOMMIT))
        json.dump({"thresholds": dict(thr, p=pre12["thresholds"]["p"], s=pre12["thresholds"]["s"]),
                   "keep": keep, "wider": WIDER, "extra": list(EXTRA), "model_sha256": _sha(MODEL),
                   "pool_sha256": _sha(POOL), "r12_model_sha256": _sha(P12.MODEL), "sense_table_sha256": _sha(R.TABLE),
                   "arm_s_model_sha256": _sha(S.MODEL), "source_sha256": {p: _sha(p) for p in SOURCES},
                   "panel_seed": SEED, "per_source": PER_SOURCE, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
                  open(PRECOMMIT, "w"), indent=1)
        print(f"pre-committed keep {keep}, thresholds {thr}")
    print("COMPLETE")


def load_m():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL) != pre["model_sha256"] or _sha(P12.MODEL) != pre["r12_model_sha256"] \
            or _sha(R.TABLE) != pre["sense_table_sha256"] or _sha(S.MODEL) != pre["arm_s_model_sha256"]:
        raise SystemExit("REFUSED: models or sense table differ from the round-14 pre-commitment")
    return pickle.load(open(MODEL, "rb")), pickle.load(open(P12.MODEL, "rb")), pre["thresholds"]


# --- fresh window ----------------------------------------------------------------------

def panel():
    """Round 12's panel rule with a new seed and 30 passages per source; excludes every document
    used in rounds 11-13 and any word 8-gram shared with their panels."""
    from cleartext.learned_words.rewrite import tokens

    if PANEL.exists():
        raise SystemExit(f"REFUSED: {PANEL} exists")
    prev = json.load(open(R.PANEL)) + json.load(open(P12.PANEL)) + json.load(open(P13.PANEL))
    used = [x["original"] for x in prev]
    for p in ("panel-r4", "panel-r5", "panel-r8"):
        used += [x["original"] for x in json.load(open(STATE / p / "passages.json"))]
    used += [s["before"] for s in R.dev_swaps()]
    grams = P12._grams(used)
    prev_docs = {(x["domain"], x["source_doc"]) for x in prev}
    out = []
    for src in ("rfc", "fedreg", "openfda"):
        docs = {}
        for line in gzip.open(R.CORPUS / f"{src}.jsonl.gz", "rt"):
            r = json.loads(line)
            docs.setdefault(r["doc"], []).append(r["s"])
        order = sorted(docs)
        rng = random.Random(f"{SEED}:{src}")
        rng.shuffle(order)
        n = 0
        for d in order:
            if n >= PER_SOURCE:
                break
            if (src, d) in prev_docs:
                continue
            sents = docs[d]
            starts = [i for i in range(len(sents) - P12.PASSAGE_SENTENCES + 1)
                      if all(R._ok_sentence(x) for x in sents[i:i + P12.PASSAGE_SENTENCES])]
            if not starts:
                continue
            i = rng.choice(starts)
            text = " ".join(sents[i:i + P12.PASSAGE_SENTENCES])
            tt = tokens(text)
            if any(tuple(tt[k:k + 8]) in grams for k in range(len(tt) - 7)):
                continue
            n += 1
            out.append({"id": f"r14-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": d,
                        "first_sentence": i, "original": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    json.dump(out, open(PANEL, "w"), indent=1)
    print(json.dumps({"passages": len(out), "by_source": Counter(p["domain"] for p in out)}, indent=1), flush=True)


def _overlap(pan):
    from cleartext.learned_words.rewrite import tokens

    from .learned_words_rewrite import pairs

    if OVERLAP.exists():
        return set(json.load(open(OVERLAP)))
    index = {}
    for p in pan:
        t = tokens(p["original"])
        for i in range(len(t) - 7):
            index.setdefault(tuple(t[i:i + 8]), set()).add(p["id"])
    hit = set()
    for _, _, a, c in pairs():
        for x in (a, c):
            t = tokens(x)
            for i in range(len(t) - 7):
                hit |= index.get(tuple(t[i:i + 8]), set())
    json.dump(sorted(hit), open(OVERLAP, "w"))
    return hit


def apply(argv):
    """Round 11's two generators (all, sense) per target, with the arm-s score and the 31 arm-s
    features of every swap (checkpointed)."""
    from cleartext.learned_words import senses as SN

    from .learned_splits import sent
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    R._threads(argv)
    load_m()
    DIR.mkdir(exist_ok=True)
    if not PANEL.exists():
        panel()
    deadline = time.monotonic() + R._seconds(argv, 1200)
    clf_s, _ = S.load_filter()
    sn = SN.load(R.TABLE)
    _, T, ids, _ = load()
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    simpler = pickle.load(open(SIMPLER, "rb"))
    pan = json.load(open(PANEL))
    skip = _overlap(pan)
    done = {r["pid"] for r in read_jsonl(APPLY)}
    todo = [p for p in pan if p["id"] not in done and p["id"] not in skip]
    print(f"{len(done)} done, {len(todo)} to do, {len(skip)} skipped for rewrite-corpus overlap", flush=True)
    with open(APPLY, "a") as f:
        for p in todo:
            if time.monotonic() > deadline:
                break
            variants, seen = R.word_swaps_two(p["original"], model, p["id"], sn)
            rec = {"pid": p["id"], "domain": p["domain"], "content_words": seen, "sentences": len(sent(p["original"]))}
            for name, swaps in variants.items():
                got, rows = [], []
                for s in swaps:
                    r = S.swap_row(model, simpler, s)
                    if r is not None:
                        got.append(s)
                        rows.append(r)
                if got:
                    X = P13._x31(T, ids, got, rows)
                    for s, r, x, sc in zip(got, rows, X, clf_s.predict_proba(X)[:, 1]):
                        s.update(margin=r["f"]["margin"], stack=float(sc), x31=[float(v) for v in x],
                                 lemma=r["lemma"], start=r["start"])
                rec[name] = [s for s in swaps if "stack" in s]
                rec[f"unresolved_{name}"] = len(swaps) - len(rec[name])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} all={len(rec['all'])} sense={len(rec['sense'])}", flush=True)
    left = len({p["id"] for p in pan} - skip - {r["pid"] for r in read_jsonl(APPLY)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


def _score(model, swaps):
    rows = [{"key": P12.pair_key(s["word"], s["rep"], s["pos"]), "stack": s["stack"], "cover": s["cover"] or 0.0,
             "x31": s["x31"], "good": 0} for s in swaps]
    return model.score(rows) if rows else np.zeros(0)


def picks(rec, m_model, p_model, thr, choose):
    """b, s, gs as before; p: the frozen round-12 arm p; m, mw: arm m at its two thresholds.
    Filters apply before the per-passage limits."""
    a = [s for s in rec["all"] if s["margin"] >= S.MIN_MARGIN]
    g = [s for s in rec["sense"] if s["margin"] >= S.MIN_MARGIN]
    for s, x, z in zip(a, _score(m_model, a), P12.score_p(p_model, a)):
        s["m"], s["p"] = float(x), float(z)
    return {"b": choose(a), "s": choose([s for s in a if s["stack"] >= thr["s"]]),
            "gs": choose([s for s in g if s["stack"] >= thr["s"]]),
            "p": choose([s for s in a if s["p"] >= thr["p"]]),
            "m": choose([s for s in a if s["m"] >= thr["m"]]),
            "mw": choose([s for s in a if s["m"] >= thr["mw"]])}


def page(argv):
    from .learned_words_apply import _skey, choose

    m_model, p_model, thr = load_m()
    items = {}
    for r in read_jsonl(APPLY):
        for arm, chosen in picks(r, m_model, p_model, thr, choose).items():
            for s in chosen:
                k = _skey(s)
                it = items.setdefault(k, dict({x: v for x, v in s.items() if x != "x31"}, key=k, arms=[],
                                              domain=r["domain"]))
                if arm not in it["arms"]:
                    it["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1214)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r14") for x in items] + [dict(x, source="old", old_mark=x["mark"])
                                                     for x in old[:P12.CONSISTENCY_ITEMS]]
    RATING.mkdir(exist_ok=True)
    for r in range(1, P12.RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r14" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    for r in range(1, P12.RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1214:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r14_items": len(items), "items": len(base),
                      "per_arm": Counter(a for x in items for a in x["arms"])}, indent=1))


def score(argv):
    from .learned_words_study import wilson

    m_model, _, _ = load_m()
    mp = json.load(open(RATING / "blind-map.json"))
    ids = {m["id"] for m in mp}
    raters = []
    for r in range(1, P12.RATERS + 1):
        marks = {x["id"]: x["mark"] for x in csv.DictReader(open(RATING / f"marks-{r}.csv"))}
        if set(marks) != ids or set(marks.values()) - set(R.SEVERITY):
            raise SystemExit(f"REFUSED: rater {r} marks are incomplete or use an unknown label")
        raters.append(marks)
    maj = {i: R.majority([m[i] for m in raters]) for i in ids}
    key = {s["key"]: s for s in json.load(open(RATING / "key.json"))["swaps"]}
    recs = read_jsonl(APPLY)
    sentences = sum(r["sentences"] for r in recs)

    def pk(m):
        s = key[m["key"]]
        return tuple(P12.pair_key(s["word"], s["rep"], s["pos"]))

    def arm_stats(marks):
        res = {}
        for arm in ARMS:
            got = [m for m in mp if m["source"] == "r14" and arm in m["arms"]]
            ms = [marks[m["id"]] for m in got]
            c = Counter(ms)
            unseen = [marks[m["id"]] for m in got if not m_model.pri.seen(pk(m))]
            res["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(c["good"], len(ms)), "meaning_rate": c["meaning"] / max(1, len(ms)),
                                 "per_100_sentences": 100 * len(ms) / max(1, sentences),
                                 "distinct_pairs": len({pk(m) for m in got}),
                                 "unseen_pairs": {"swaps": len(unseen), "good": unseen.count("good")},
                                 "by_domain": {d: {"swaps": sum(key[m["key"]]["domain"] == d for m in got),
                                                   "good": sum(key[m["key"]]["domain"] == d and marks[m["id"]] == "good"
                                                               for m in got)} for d in ("rfc", "fedreg", "openfda")}}
        return res

    out = {"items": len(ids), "raters": P12.RATERS, "passages": len(recs), "sentences": sentences,
           "majority": arm_stats(maj), "per_rater": [arm_stats(m) for m in raters]}
    M = out["majority"]
    out["yield_vs_b"] = {a: M["arm_" + a]["swaps"] / max(1, M["arm_b"]["swaps"]) for a in ARMS}
    m = M["arm_m"]
    out["success"] = bool(m["good_rate"] >= 0.90 and m["meaning_rate"] <= 0.08 and out["yield_vs_b"]["m"] >= 0.40)
    r14 = [x["id"] for x in mp if x["source"] == "r14"]
    out["unanimous_share"] = sum(len({rt[i] for rt in raters}) == 1 for i in r14) / max(1, len(r14))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r14) / max(1, len(r14))
                                 for a in range(P12.RATERS) for c in range(a + 1, P12.RATERS)}
    old = [x for x in mp if x["source"] == "old"]
    out["consistency_with_old"] = [sum(x["old_mark"] == rt[x["id"]] for x in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"train": train, "apply": apply, "page": page, "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r14 train|apply|page|score")
