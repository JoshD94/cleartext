"""Round 26: arm f refitted on every rated row through round 25, and an unseen-pair threshold
calibrated on recent windows (see ROUND26.md).

  learned-words r26 train                    CV gate; final fits and pre-commit if go
  learned-words r26 apply [--max-seconds N]  fresh panel (once), round 24's apply (checkpointed)
  learned-words r26 page | score             three blind rater files; majority-vote scoring

Raw text, pools and marks stay in the state directory (r26/).
"""

import contextlib
import csv
import hashlib
import json
import pickle
import random
import time
from collections import Counter

import numpy as np

from . import learned_words_r12 as P12
from . import learned_words_r14 as P14
from . import learned_words_r19 as P19
from . import learned_words_r21 as Q
from . import learned_words_r22 as G
from . import learned_words_r24 as F
from . import learned_words_r25 as F25
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r26"
POOL = DIR / "pool.json"
MODEL_F2 = DIR / "model-f2.pkl"
PANEL = DIR / "passages.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r26-precommit.json"
CV_REPORT = REPORT / "r26-cv.json"
RESULT = REPORT / "round26.json"
SEED = 20261226
PER_SOURCE = {"news": ("news2", 45), "wiki": ("wiki2", 55), "openfda": ("openfda", 45)}
GENRES, EXT_NAMES = F.GENRES, F.EXT_NAMES
RECENT = ("r17", "r18", "r19", "r20", "r21", "r22", "r24", "r25")  # three raters, round 11's generator, current pool era
UNSEEN_PRECISION = 0.90  # f2u: target Good rate of recent unseen-pair arm-b rows above the unseen threshold
UNSEEN_MIN_KEPT = 20
ARMS = ("b", "p50", "f", "f2", "f2u")
REPEATS = 10
SOURCES = ("evals/learned_words_r26.py",) + F25.SOURCES
_docs = F._docs


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# --- pool ------------------------------------------------------------------------------

def _rated(rating, apply_path, tag):
    """A round-24-style window's rated swaps, with majority marks and every arm-f input."""
    from . import learned_words_r9 as R
    from .learned_words_apply import _skey

    mp = json.load(open(rating / "blind-map.json"))
    raters = [{x["id"]: x["mark"] for x in csv.DictReader(open(rating / f"marks-{r}.csv"))}
              for r in range(1, P12.RATERS + 1)]
    sw = {}
    for rec in read_jsonl(apply_path):
        for s in rec.get("all", []):
            sw[_skey(s)] = s
    key = {s["key"]: s for s in json.load(open(rating / "key.json"))["swaps"]}
    out = []
    for m in mp:
        if m["source"] != tag:
            continue
        s, a = key[m["key"]], sw[m["key"]]
        mark = R.majority([rt[m["id"]] for rt in raters])
        k = list(P12.pair_key(s["word"], s["rep"], s["pos"]))
        out.append({"src": f"{tag}:{m['key']}", "round": tag, "group": f"{tag}:{s['pid']}", "word": s["word"],
                    "rep": s["rep"], "pos": s["pos"], "mark": mark, "good": int(mark == "good"), "b": "b" in m["arms"],
                    "stack": float(a["stack"]), "cover": float(a["cover"] or 0.0), "x31": a["x31"], "runner_up": False,
                    "key": k, "before": a["before"], "after": a["after"], "domain": s["domain"],
                    "ctx": Q.context(a["before"], a["after"], a["rep"]), "tech": s["domain"] not in GENRES,
                    "genre": s["domain"] in GENRES, "fr": a["fr"], "nb": [tuple(x) for x in a["nb"]]})
    return out


def pool():
    """Round 24's 1,454 rows (with frame and lexicon inputs) plus rounds 24 and 25's rated swaps."""
    from cleartext.learned_words.pairlex import PairLex

    lexicon = PairLex()
    rows = F.attach(json.load(open(F.POOL)), lexicon)
    rows += _rated(F.RATING, F.APPLY, "r24") + _rated(F25.RATING, F25.APPLY, "r25")
    for r in rows:
        r["lex"] = lexicon.features(r["word"], r["rep"], r["key"])
    return rows


def _refs(rows):
    b = np.array([r["b"] for r in rows])
    tech = b & np.array([r["round"] in Q.TECH or (r["round"] in ("r21", "r24", "r25") and not r["genre"]) for r in rows])
    gen = b & np.array([bool(r.get("genre")) for r in rows])
    recent = b & np.isin([r["round"] for r in rows], RECENT)
    return tech, gen, recent


def unseen_cut(scores, good, t):
    """The lowest score whose recent unseen-pair arm-b rows at or above it are >= UNSEEN_PRECISION Good,
    with at least UNSEEN_MIN_KEPT kept; never stricter than the seen threshold t."""
    o = np.argsort(-scores)
    ss, yy = scores[o], good[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    ok = [k for k in range(UNSEEN_MIN_KEPT - 1, len(yy)) if prec[k] >= UNSEEN_PRECISION]
    return min(t, float(ss[ok[-1]])) if ok else t


def _unseen_inner(groups_tr, keyp_tr, seed):
    u = np.zeros(len(groups_tr), bool)
    for itr, ite in P12._folds(groups_tr, seed):
        sp = set(keyp_tr[itr])
        u[ite] = [k not in sp for k in keyp_tr[ite]]
    return u


def nested(rows, groups, seed):
    """Passage-grouped nested CV of arm f's class: the seen threshold keeps F.KEEP_TECH of the training
    folds' technical arm-b rows by inner out-of-fold score; f2u's unseen threshold is unseen_cut on the
    training folds' recent arm-b rows whose pair is unseen in their inner training folds."""
    tech, _, recent = _refs(rows)
    keyp = np.array(["|".join(r["key"]) for r in rows])
    y = np.array([r["good"] for r in rows])
    k2, ku = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    seen = np.zeros(len(rows), bool)
    for tr, te in P12._folds(groups, seed):
        sub = [rows[i] for i in tr]
        inner = P14.oof(F.make_f, sub, groups[tr], seed + 1)
        t = float(np.quantile(inner[tech[tr]], 1 - F.KEEP_TECH))
        iu = _unseen_inner(groups[tr], keyp[tr], seed + 1) & recent[tr]
        tu = unseen_cut(inner[iu], y[tr][iu], t)
        sc = F.make_f(sub).score([rows[i] for i in te])
        sp = set(keyp[tr])
        seen[te] = [k in sp for k in keyp[te]]
        k2[te] = sc >= t
        ku[te] = np.where(seen[te], sc >= t, sc >= tu)
    return k2, ku, seen


def _stats(kp, seen, marks, ref):
    c = Counter(marks[kp & ref])
    u = kp & ref & ~seen
    return {"kept": int((kp & ref).sum()), "keep_share": float((kp & ref).sum() / max(1, ref.sum())),
            "good_rate": c["good"] / max(1, (kp & ref).sum()), "meaning": int(c["meaning"]),
            "unseen_rows": int((ref & ~seen).sum()), "unseen_kept": int(u.sum()), "unseen_good": int((marks[u] == "good").sum())}


def train(argv):
    from cleartext.learned_words import pairlex as PL

    from . import learned_words_r9 as R

    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    DIR.mkdir(parents=True, exist_ok=True)
    rows = pool()
    json.dump(rows, open(POOL, "w"))
    marks = np.array([r["mark"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    tech, gen, recent = _refs(rows)
    res = {"rows": len(rows), "tech_ref": int(tech.sum()), "genre_ref": int(gen.sum()), "recent_ref": int(recent.sum()),
           "recent_rounds": list(RECENT), "unseen_precision": UNSEEN_PRECISION, "unseen_min_kept": UNSEEN_MIN_KEPT,
           "repeats": REPEATS, "f_args": F.F_ARGS}
    runs = []
    for k in range(REPEATS):
        k2, ku, seen = nested(rows, groups, P12.SEED + 10 * k)
        runs.append({f"{arm}_{part}": _stats(kp, seen, marks, ref) for arm, kp in (("f2", k2), ("f2u", ku))
                     for part, ref in (("tech", tech), ("recent", recent), ("genre", gen))})
    res["cv"] = {k: F._mean([r[k] for r in runs]) for k in runs[0]}
    c = res["cv"]
    res["go_conditions"] = {
        "f2_tech_good_ge_0.92": c["f2_tech"]["good_rate"] >= 0.92,
        "f2_tech_keep_ge_0.45": c["f2_tech"]["keep_share"] >= 0.45,
        "f2u_recent_unseen_kept_ge_1.3x_f2": c["f2u_recent"]["unseen_kept"] >= 1.3 * c["f2_recent"]["unseen_kept"],
        "f2u_recent_unseen_good_ge_0.88": c["f2u_recent"]["unseen_good"] >= 0.88 * max(1, c["f2u_recent"]["unseen_kept"]),
        "f2u_recent_good_ge_0.91": c["f2u_recent"]["good_rate"] >= 0.91}
    res["f2_go"] = bool(c["f2_tech"]["good_rate"] >= 0.92 and c["f2_tech"]["keep_share"] >= 0.45)
    res["go"] = bool(all(res["go_conditions"].values()))
    res["pool_sha256"] = _sha(POOL)
    res["source_sha256"] = {p: _sha(p) for p in SOURCES}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "source_sha256"}, indent=1))
    if not res["f2_go"]:
        print("NO-GO: no fresh text is spent")
        return
    keyp = np.array(["|".join(r["key"]) for r in rows])
    y = np.array([r["good"] for r in rows])
    oofs, unseen = [], np.zeros(len(rows))
    for k in range(REPEATS):
        oofs.append(P14.oof(F.make_f, rows, groups, P12.SEED + k))
        for tr, te in P12._folds(groups, P12.SEED + k):
            sp = set(keyp[tr])
            unseen[te] += [x not in sp for x in keyp[te]]
    full = np.mean(oofs, axis=0)
    t = float(np.quantile(full[tech], 1 - F.KEEP_TECH))
    u = recent & (unseen / REPEATS >= 0.5)
    tu = unseen_cut(full[u], y[u], t)
    pickle.dump(F.make_f(rows), open(MODEL_F2, "wb"))
    pre24 = json.load(open(F.PRECOMMIT))
    arms = [a for a in ARMS if a != "f2u" or res["go"]]
    json.dump({"thresholds": {"f2": t, "f2u_unseen": tu, "f": pre24["thresholds"]["f"], "p50": pre24["thresholds"]["p50"]},
               "arms": arms, "go": res["go"], "f2_go": res["f2_go"], "f_args": F.F_ARGS,
               "model_f2_sha256": _sha(MODEL_F2), "model_f_sha256": _sha(F.MODEL_F), "model_p50_sha256": _sha(P19.MODEL),
               "model_qg_sha256": _sha(G.MODEL_QG), "pool_sha256": _sha(POOL),
               "pair_lexical_sha256": _sha(PL.LEXICAL), "wc_lexicon_sha256": _sha(PL.RAW_WC),
               "sense_table_sha256": _sha(R.TABLE), "arm_s_model_sha256": _sha(S.MODEL),
               "source_sha256": {p: _sha(p) for p in SOURCES}, "panel_seed": SEED,
               "per_source": {k: v[1] for k, v in PER_SOURCE.items()},
               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
    print(f"pre-committed thresholds f2={t:.4f} f2u_unseen={tu:.4f}; arms {arms}")
    print("COMPLETE")


# --- fresh window ----------------------------------------------------------------------

def load_models():
    pre = json.load(open(PRECOMMIT))
    from cleartext.learned_words import pairlex as PL

    from . import learned_words_r9 as R

    for path, k in ((MODEL_F2, "model_f2_sha256"), (F.MODEL_F, "model_f_sha256"), (P19.MODEL, "model_p50_sha256"),
                    (G.MODEL_QG, "model_qg_sha256"), (R.TABLE, "sense_table_sha256"), (S.MODEL, "arm_s_model_sha256"),
                    (PL.LEXICAL, "pair_lexical_sha256"), (PL.RAW_WC, "wc_lexicon_sha256")):
        if _sha(path) != pre[k]:
            raise SystemExit(f"REFUSED: {path.name} differs from the round-26 pre-commitment")
    return {"f2": pickle.load(open(MODEL_F2, "rb")), "f": pickle.load(open(F.MODEL_F, "rb")),
            "p50": pickle.load(open(P19.MODEL, "rb"))}, pre


def picks(rec, models, thr, choose, lexicon):
    """b; p50 and round 24's f, frozen; f2 at its seen threshold; f2u: f2 with the unseen threshold for
    pairs absent from f2's pool. Filters apply before the per-passage limits."""
    a = [s for s in rec["all"] if s["margin"] >= S.MIN_MARGIN]
    if a:
        for s, x in zip(a, P12.score_p(models["p50"], a)):
            s["sc_p50"] = float(x)
        rows = F.swap_rows(a, lexicon)
        for s, x in zip(a, models["f"].score(rows)):
            s["sc_f"] = float(x)
        for s, r, x in zip(a, rows, models["f2"].score(rows)):
            s["sc_f2"] = float(x)
            s["seen2"] = bool(models["f2"].pri.seen(r["key"]))
    return {"b": choose(a), "p50": choose([s for s in a if s["sc_p50"] >= thr["p50"]]),
            "f": choose([s for s in a if s["sc_f"] >= thr["f"]]),
            "f2": choose([s for s in a if s["sc_f2"] >= thr["f2"]]),
            "f2u": choose([s for s in a if s["sc_f2"] >= (thr["f2"] if s["seen2"] else thr["f2u_unseen"])])}


def panel():
    """Round 25's panel rule (seed SEED) on the extended genre corpora and unused openFDA labels.
    Excludes every document of every earlier window (rounds 11-25, both Joshua windows) and any
    passage sharing a word 8-gram with any earlier panel or window, any rated development sentence
    or test-set v3's panel file (exclusion only)."""
    from cleartext.learned_words.rewrite import tokens

    from . import learned_words_r9 as R

    if PANEL.exists():
        raise SystemExit(f"REFUSED: {PANEL} exists")
    prev = []
    for p in (R.PANEL, Q.P12.PANEL, Q.P13.PANEL, Q.P17.PANEL, Q.P18.PANEL, Q.P19.PANEL, Q.P20.PANEL,
              Q.DIR / "passages.json", Q.JDIR / "passages.json", STATE / "joshua-r21b/passages.json", G.PANEL, F.PANEL, F25.PANEL):
        prev += json.load(open(p))
    used = [x["original"] for x in prev]
    for p in ("panel-r4", "panel-r5", "panel-r8"):
        used += [x["original"] for x in json.load(open(STATE / p / "passages.json"))]
    used += [s["before"] for s in R.dev_swaps()]
    used += [x["original"] for x in json.load(open(Q.V3_PANEL))]
    grams = P12._grams(used)
    prev_docs = {(x["domain"], x["source_doc"]) for x in prev}
    out = []
    for dom, (name, want) in PER_SOURCE.items():
        docs = _docs(name)
        order = sorted(docs)
        rng = random.Random(f"{SEED}:{dom}")
        rng.shuffle(order)
        n = 0
        for doc in order:
            if n >= want:
                break
            if (dom, doc) in prev_docs:
                continue
            sents = docs[doc]
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
            out.append({"id": f"r26-{dom}-{n:02d}", "domain": dom, "kind": "standard", "source_doc": doc,
                        "corpus": name, "first_sentence": i, "original": text,
                        "sha256": hashlib.sha256(text.encode()).hexdigest()})
    DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(PANEL, "w"), indent=1)
    print(json.dumps({"passages": len(out), "by_source": Counter(p["domain"] for p in out)}), flush=True)


@contextlib.contextmanager
def window():
    names = ("DIR", "PANEL", "APPLY", "RATING", "PRECOMMIT", "RESULT", "SEED", "panel", "load_models")
    saved = {k: getattr(F, k) for k in names}
    F.DIR, F.PANEL, F.APPLY, F.RATING = DIR, PANEL, APPLY, RATING
    F.PRECOMMIT, F.RESULT, F.SEED, F.panel, F.load_models = PRECOMMIT, RESULT, SEED, panel, load_models
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(F, k, v)


def page(argv):
    from cleartext.learned_words.pairlex import PairLex

    from .learned_words_apply import _skey, choose

    models, pre = load_models()
    lexicon = PairLex()
    items = {}
    for r in read_jsonl(APPLY):
        for arm, chosen in picks(r, models, pre["thresholds"], choose, lexicon).items():
            if arm not in pre["arms"]:
                continue
            for s in chosen:
                k = _skey(s)
                it = items.setdefault(k, dict({x: v for x, v in s.items() if x not in ("x31", "fr", "nb")}, key=k, arms=[],
                                              domain=r["domain"], shown=Q.realized(s)))
                if arm not in it["arms"]:
                    it["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1226)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r26") for x in items] + [dict(x, source="old", old_mark=x["mark"], shown=x["after"])
                                                     for x in old[:P12.CONSISTENCY_ITEMS]]
    RATING.mkdir(parents=True, exist_ok=True)
    for r in range(1, P12.RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r26" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    brief = (STATE / "r9/rater-brief.txt").read_text()
    for r in range(1, P12.RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1226:{r}").shuffle(order)
        path = RATING / f"blind-{r}.json"
        json.dump([{"id": ids[n], "before": x["before"], "after": x["shown"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(path, "w"), indent=1)
        n = len(base)
        b = brief.replace(str(STATE / "rating-r5/blind-mixed.json"), str(path)).replace("254 items", f"{n} items")
        b = b.replace("(all 254, ids m1..m254)", f"(all {n}, ids m1..m{n})")
        b = b.replace(str(STATE / "rating-r5/blind-mixed-marks.csv"), str(RATING / f"marks-{r}.csv"))
        b = b.replace("has 255 lines", f"has {n + 1} lines")
        (RATING / f"brief-{r}.txt").write_text(b)
    print(json.dumps({"r26_items": len(items), "items": len(base), "per_arm": Counter(a for x in items for a in x["arms"]),
                      "per_domain": Counter(x["domain"] for x in items)}, indent=1))


def score(argv):
    from . import learned_words_r9 as R
    from .learned_words_study import wilson

    _, pre = load_models()
    arms = pre["arms"]
    pool_rows = json.load(open(POOL))
    pool_pairs = {tuple(r["key"]) for r in pool_rows}
    pool24 = {tuple(r["key"]) for r in json.load(open(F.POOL))}
    c = {}
    for r in pool_rows:
        x = c.setdefault(tuple(r["key"]), [0, 0])
        x[0] += r["good"]
        x[1] += 1
    well = {k for k, (g, n) in c.items() if n >= 3 and g / n >= 0.7}
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
        for arm in arms:
            got = [m for m in mp if m["source"] == "r26" and arm in m["arms"]]
            ms = [marks[m["id"]] for m in got]
            cc = Counter(ms)
            un = [marks[m["id"]] for m in got if pk(m) not in pool_pairs]
            un24 = [marks[m["id"]] for m in got if pk(m) not in pool24]
            fw = [marks[m["id"]] for m in got if pk(m) in well]
            res["arm_" + arm] = {"swaps": len(ms), **cc, "good_rate": cc["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(cc["good"], len(ms)), "meaning_rate": cc["meaning"] / max(1, len(ms)),
                                 "per_100_sentences": 100 * len(ms) / max(1, sentences),
                                 "distinct_pairs": len({pk(m) for m in got}),
                                 "unseen_pairs": {"swaps": len(un), "good": un.count("good"), "meaning": un.count("meaning")},
                                 "unseen_vs_1454": {"swaps": len(un24), "good": un24.count("good")},
                                 "well_rated_pairs": {"swaps": len(fw), "meaning": fw.count("meaning")},
                                 "by_domain": {d: {"swaps": sum(key[m["key"]]["domain"] == d for m in got),
                                                   "good": sum(key[m["key"]]["domain"] == d and marks[m["id"]] == "good"
                                                               for m in got)} for d in PER_SOURCE}}
        return res

    out = {"items": len(ids), "raters": P12.RATERS, "passages": len(recs), "sentences": sentences,
           "majority": arm_stats(maj), "per_rater": [arm_stats(m) for m in raters]}
    M = out["majority"]
    out["yield_vs_b"] = {a: M["arm_" + a]["swaps"] / max(1, M["arm_b"]["swaps"]) for a in arms}
    for arm in ("f2", "f2u"):
        if arm in arms:
            v = M["arm_" + arm]
            out["success_" + arm] = {"good_ge_0.90": v["good_rate"] >= 0.90, "meaning_le_0.08": v["meaning_rate"] <= 0.08,
                                     "yield_ge_0.40": out["yield_vs_b"][arm] >= 0.40}
    if "f2u" in arms:
        v, w = M["arm_f2u"], M["arm_f2"]
        out["unseen_goal_f2u"] = {"unseen_swaps_ge_1.3x_f2": v["unseen_pairs"]["swaps"] >= 1.3 * w["unseen_pairs"]["swaps"],
                                  "unseen_good_ge_0.90": v["unseen_pairs"]["good"] >= 0.90 * max(1, v["unseen_pairs"]["swaps"])}
    r26 = [x["id"] for x in mp if x["source"] == "r26"]
    out["unanimous_share"] = sum(len({rt[i] for rt in raters}) == 1 for i in r26) / max(1, len(r26))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r26) / max(1, len(r26))
                                 for a in range(P12.RATERS) for c in range(a + 1, P12.RATERS)}
    old = [x for x in mp if x["source"] == "old"]
    out["consistency_with_old"] = [sum(x["old_mark"] == rt[x["id"]] for x in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_rater"}, indent=1))


def main(argv):
    if argv[:1] == ["train"]:
        return train(argv[1:])
    if argv[:1] == ["apply"]:
        with window():
            return F.apply(argv[1:])
    cmd = {"page": page, "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r26 train|apply|page|score")
