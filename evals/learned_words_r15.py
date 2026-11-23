"""Round 15: arm pv, arm p refitted on every rated row at a wider keep, then a corpus
collocation veto (see ROUND15.md).

  learned-words r15 train                    CV gate; final fit and pre-commit if go
  learned-words r15 apply [--max-seconds N]  fresh panel (once) and round 11's two generators (checkpointed)
  learned-words r15 page | score             three blind rater files; majority-vote scoring

The veto feature is round 6's `max_drop` (context_fit.py): the largest drop, target to
replacement, in corpus association over the frames around the swap. Counts come from the
round-6 technical corpus with the swap's whole source document removed (only its own
sentence for text from outside the corpus). Raw text, pools and marks stay in r15/.
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
from . import learned_words_r14 as P14
from . import learned_words_r9 as R
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r15"
POOL = DIR / "pool.json"
MODEL = DIR / "model.pkl"
PANEL = DIR / "passages.json"
OVERLAP = DIR / "panel-overlap.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r15-precommit.json"
CV_REPORT = REPORT / "r15-cv.json"
RESULT = REPORT / "round15.json"
SEED = 20261215
KEEP = 0.60  # arm p's keep before the veto
KEEP_P = 0.40  # comparator: arm p alone at about the yield arm pv is expected to reach
GOOD_KEPT = 0.80  # veto cut-off: the 80th percentile of max_drop among Good technical-window rows
REPEATS = 10
PER_SOURCE = 30
ARMS = ("b", "s", "gs", "p", "p40", "pv")
TECH = ("r11", "r12")
SOURCES = ("evals/learned_words_r15.py", "evals/learned_words_r14.py", "evals/learned_words_r13.py",
           "models/src/cleartext/learned_words/context_fit.py", "evals/learned_words_corpus.py") + P12.SOURCES


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# --- the veto feature ----------------------------------------------------------------

def doc_ngrams(keys):
    """{(source, doc): (unigram, bigram, trigram) Counters} over each document's corpus sentences."""
    from .learned_words_corpus import tokens

    out = {}
    for src in ("rfc", "fedreg", "openfda"):
        for line in gzip.open(R.CORPUS / f"{src}.jsonl.gz", "rt"):
            r = json.loads(line)
            k = (src, r["doc"])
            if k in keys:
                u, b, t = out.setdefault(k, (Counter(), Counter(), Counter()))
                tt = tokens(r["s"])
                u.update(tt)
                b.update(zip(tt, tt[1:]))
                t.update(zip(tt, tt[1:], tt[2:]))
    return out


def max_drop(before, word, rep, start, own=None):
    """Round 6's max_drop for one swap; `own` = n-gram Counters to remove (default: the sentence)."""
    from cleartext.learned_words import context_fit as CF

    from .learned_words_contextfit import COUNTS
    from .learned_words_corpus import tokens

    toks = tokens(before)
    w = (tokens(word) or [word.lower()])[0]
    i = len(tokens(before[:start])) if start is not None else -1
    if not (0 <= i < len(toks)) or toks[i] != w:
        i = toks.index(w) if w in toks else 0
    C = CF.Counts(CF.load_counts(COUNTS), own or CF.own_ngrams(toks))
    return float(CF.features(C, toks, i, (tokens(rep) or [rep.lower()])[0])["max_drop"])


def _panel_docs():
    pans = [R.PANEL, P12.PANEL, P13.PANEL] + ([PANEL] if PANEL.exists() else [])
    return {p["id"]: (p["domain"], p["source_doc"]) for path in pans for p in json.load(open(path))}


def pool():
    """Round 14's frozen 620-row pool (rounds 3-8, 11 and 12) with the veto feature."""
    rows = json.load(open(P14.POOL))
    sent = {s["src"]: s for s in R.dev_swaps()}
    sent.update({"r11:" + s["key"]: s for s in json.load(open(R.RATING11 / "key.json"))["swaps"]})
    sent.update({"r12:" + s["key"]: s for s in json.load(open(P12.RATING / "key.json"))["swaps"]})
    pdoc = _panel_docs()
    docs = doc_ngrams({pdoc[sent[r["src"]]["pid"]] for r in rows if r["round"] in TECH})
    for r in rows:
        s = sent[r["src"]]
        own = docs[pdoc[s["pid"]]] if r["round"] in TECH else None
        r["drop"] = max_drop(s["before"], s["word"], s["rep"], s.get("start"), own)
        r["tech"] = r["round"] in TECH
    return rows


def cutoff(rows):
    """The veto cut-off learned from rated rows: GOOD_KEPT quantile of max_drop on Good technical rows."""
    d = np.array([r["drop"] for r in rows if r["tech"] and r["good"]])
    return float(np.quantile(d, GOOD_KEPT))


# --- gate ----------------------------------------------------------------------------

def nested(rows, groups, ref, seed):
    """Outer out-of-fold keep decisions for arm pv and for arm p alone at KEEP_P."""
    n = len(rows)
    kv, kp = np.zeros(n, bool), np.zeros(n, bool)
    drop = np.array([r["drop"] for r in rows])
    for tr, te in P12._folds(groups, seed):
        sub = [rows[i] for i in tr]
        inner = P14.oof(P12.ArmP, sub, groups[tr], seed + 1)
        t60 = np.quantile(inner[ref[tr]], 1 - KEEP)
        t40 = np.quantile(inner[ref[tr]], 1 - KEEP_P)
        tau = cutoff(sub)
        sc = P12.ArmP(sub).score([rows[i] for i in te])
        kv[te] = (sc >= t60) & (drop[te] <= tau)
        kp[te] = sc >= t40
    return kv, kp


def train(argv):
    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    DIR.mkdir(exist_ok=True)
    rows = pool()
    marks = np.array([r["mark"] for r in rows])
    b = np.array([r["b"] for r in rows])
    rnd = np.array([r["round"] for r in rows])
    ref = b & np.isin(rnd, TECH)
    ref12 = ref & (rnd == "r12")
    if len(rows) != 620 or ref.sum() != 233:
        raise SystemExit(f"REFUSED: pool has {len(rows)} rows, {ref.sum()} round-11/12 arm-b rows")
    json.dump(rows, open(POOL, "w"))
    groups = np.array([r["group"] for r in rows])
    res = {"rows": len(rows), "ref_rows": int(ref.sum()), "keep": KEEP, "keep_p": KEEP_P, "good_kept": GOOD_KEPT,
           "repeats": REPEATS, "cutoff_all_rows": cutoff(rows)}

    # A. passage-grouped nested CV on the rounds 11-12 arm-b rows
    runs = []
    for k in range(REPEATS):
        kv, kp = nested(rows, groups, ref, P12.SEED + 10 * k)
        runs.append({"pv": P12._stats(kv, marks, ref), "p40": P12._stats(kp, marks, ref),
                     "pv_r12": P12._stats(kv, marks, ref12), "p40_r12": P12._stats(kp, marks, ref12)})
    res["passage_cv"] = {a: {f: float(np.mean([r[a][f] for r in runs]))
                             for f in ("kept", "keep_share", "good_rate", "meaning", "notsimpler")}
                         for a in ("pv", "p40", "pv_r12", "p40_r12")}
    res["passage_cv"]["paired_gain"] = float(np.mean([r["pv"]["good_rate"] - r["p40"]["good_rate"] for r in runs]))

    # B. time split: fit on rounds 3-11, thresholds and cut-off from them, score round 12
    early = rnd != "r12"
    e_rows = [r for r, x in zip(rows, early) if x]
    l_rows = [r for r, x in zip(rows, ~early) if x]
    inner = np.mean([P14.oof(P12.ArmP, e_rows, groups[early], P12.SEED + k) for k in range(5)], axis=0)
    t60 = np.quantile(inner[ref[early]], 1 - KEEP)
    t40 = np.quantile(inner[ref[early]], 1 - KEEP_P)
    tau = cutoff(e_rows)
    sc = np.zeros(len(rows))
    sc[~early] = P12.ArmP(e_rows).score(l_rows)
    drop = np.array([r["drop"] for r in rows])
    res["time_split_r12"] = {"pv": P12._stats((~early) & (sc >= t60) & (drop <= tau), marks, ref12),
                             "p40": P12._stats((~early) & (sc >= t40), marks, ref12), "cutoff": tau}
    pc, ts = res["passage_cv"], res["time_split_r12"]
    res["go_conditions"] = {"cv_good_ge_0.92": pc["pv"]["good_rate"] >= 0.92,
                            "cv_gain_vs_p40_ge_0.03": pc["paired_gain"] >= 0.03,
                            "cv_keep_ge_0.38": pc["pv"]["keep_share"] >= 0.38,
                            "time_split_good_ge_0.90_keep_ge_0.35": ts["pv"]["good_rate"] >= 0.90
                            and ts["pv"]["keep_share"] >= 0.35}
    res["go"] = bool(all(res["go_conditions"].values()))
    res["pool_sha256"] = _sha(POOL)
    res["source_sha256"] = {p: _sha(p) for p in SOURCES}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "source_sha256"}, indent=1))
    if res["go"]:
        full = np.mean([P14.oof(P12.ArmP, rows, groups, P12.SEED + k) for k in range(REPEATS)], axis=0)
        thr = {"pv": float(np.quantile(full[ref], 1 - KEEP)), "p40": float(np.quantile(full[ref], 1 - KEEP_P)),
               "cutoff": cutoff(rows)}
        pickle.dump(P12.ArmP(rows), open(MODEL, "wb"))
        pre12 = json.load(open(P12.PRECOMMIT))
        json.dump({"thresholds": dict(thr, p=pre12["thresholds"]["p"], s=pre12["thresholds"]["s"]),
                   "keep": KEEP, "keep_p": KEEP_P, "good_kept": GOOD_KEPT, "model_sha256": _sha(MODEL),
                   "pool_sha256": _sha(POOL), "r12_model_sha256": _sha(P12.MODEL), "sense_table_sha256": _sha(R.TABLE),
                   "arm_s_model_sha256": _sha(S.MODEL), "counts_sha256": _sha(_counts()),
                   "source_sha256": {p: _sha(p) for p in SOURCES}, "panel_seed": SEED, "per_source": PER_SOURCE,
                   "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
        print(f"pre-committed thresholds {thr}")
    print("COMPLETE")


def _counts():
    from .learned_words_contextfit import COUNTS

    return COUNTS


def load_v():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL) != pre["model_sha256"] or _sha(P12.MODEL) != pre["r12_model_sha256"] \
            or _sha(R.TABLE) != pre["sense_table_sha256"] or _sha(S.MODEL) != pre["arm_s_model_sha256"] \
            or _sha(_counts()) != pre["counts_sha256"]:
        raise SystemExit("REFUSED: models, counts or sense table differ from the round-15 pre-commitment")
    return pickle.load(open(MODEL, "rb")), pickle.load(open(P12.MODEL, "rb")), pre["thresholds"]


# --- fresh window ----------------------------------------------------------------------

def panel():
    """Round 14's panel rule (30 passages per source) with a new seed; excludes every document
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
            out.append({"id": f"r15-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": d,
                        "first_sentence": i, "original": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    json.dump(out, open(PANEL, "w"), indent=1)
    print(json.dumps({"passages": len(out), "by_source": Counter(p["domain"] for p in out)}, indent=1), flush=True)


def apply(argv):
    """Round 11's two generators (all, sense) per target, with the arm-s score and the 31 arm-s
    features of every swap (checkpointed). Mirrors round 14's apply."""
    from cleartext.learned_words import senses as SN

    from .learned_splits import sent
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    R._threads(argv)
    load_v()
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


def picks(rec, v_model, p_model, thr, choose, docs):
    """b, s, gs as before; p: the frozen round-12 arm p; p40: the refitted arm p at a 40% keep;
    pv: the refitted arm p at a 60% keep, then the collocation veto. Filters apply before the
    per-passage limits."""
    a = [s for s in rec["all"] if s["margin"] >= S.MIN_MARGIN]
    g = [s for s in rec["sense"] if s["margin"] >= S.MIN_MARGIN]
    for s, x, z in zip(a, P12.score_p(v_model, a), P12.score_p(p_model, a)):
        s["v"], s["p"] = float(x), float(z)
        s["drop"] = max_drop(s["before"], s["word"], s["rep"], s.get("start"), docs.get(s["pid"]))
    return {"b": choose(a), "s": choose([s for s in a if s["stack"] >= thr["s"]]),
            "gs": choose([s for s in g if s["stack"] >= thr["s"]]),
            "p": choose([s for s in a if s["p"] >= thr["p"]]),
            "p40": choose([s for s in a if s["v"] >= thr["p40"]]),
            "pv": choose([s for s in a if s["v"] >= thr["pv"] and s["drop"] <= thr["cutoff"]])}


def _docs_for_panel():
    pan = json.load(open(PANEL))
    ng = doc_ngrams({(p["domain"], p["source_doc"]) for p in pan})
    return {p["id"]: ng[(p["domain"], p["source_doc"])] for p in pan}


def page(argv):
    from .learned_words_apply import _skey, choose

    v_model, p_model, thr = load_v()
    docs = _docs_for_panel()
    items = {}
    for r in read_jsonl(APPLY):
        for arm, chosen in picks(r, v_model, p_model, thr, choose, docs).items():
            for s in chosen:
                k = _skey(s)
                it = items.setdefault(k, dict({x: v for x, v in s.items() if x != "x31"}, key=k, arms=[],
                                              domain=r["domain"]))
                if arm not in it["arms"]:
                    it["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1215)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r15") for x in items] + [dict(x, source="old", old_mark=x["mark"])
                                                     for x in old[:P12.CONSISTENCY_ITEMS]]
    RATING.mkdir(exist_ok=True)
    for r in range(1, P12.RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r15" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    for r in range(1, P12.RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1215:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r15_items": len(items), "items": len(base),
                      "per_arm": Counter(a for x in items for a in x["arms"])}, indent=1))


def score(argv):
    from .learned_words_study import wilson

    v_model, _, _ = load_v()
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
            got = [m for m in mp if m["source"] == "r15" and arm in m["arms"]]
            ms = [marks[m["id"]] for m in got]
            c = Counter(ms)
            unseen = [marks[m["id"]] for m in got if not v_model.pri.seen(pk(m))]
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
    v = M["arm_pv"]
    out["success"] = bool(v["good_rate"] >= 0.90 and v["meaning_rate"] <= 0.08 and out["yield_vs_b"]["pv"] >= 0.40)
    r15 = [x["id"] for x in mp if x["source"] == "r15"]
    out["unanimous_share"] = sum(len({rt[i] for rt in raters}) == 1 for i in r15) / max(1, len(r15))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r15) / max(1, len(r15))
                                 for a in range(P12.RATERS) for c in range(a + 1, P12.RATERS)}
    old = [x for x in mp if x["source"] == "old"]
    out["consistency_with_old"] = [sum(x["old_mark"] == rt[x["id"]] for x in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"train": train, "apply": apply, "page": page, "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r15 train|apply|page|score")
