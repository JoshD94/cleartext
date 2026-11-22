"""Round 16: round 15's arm pv, gated on round 13's marks (held out from every design step),
then refitted with them (see ROUND16.md).

  learned-words r16 train                    gate on round 13; final fit and pre-commit if go
  learned-words r16 apply [--max-seconds N]  fresh panel (once) and round 11's two generators (checkpointed)
  learned-words r16 page | score             three blind rater files; majority-vote scoring

Raw text, pools and marks stay in the state directory (r16/).
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
from . import learned_words_r15 as P15
from . import learned_words_r9 as R
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r16"
POOL = DIR / "pool.json"
MODEL = DIR / "model.pkl"
PANEL = DIR / "passages.json"
OVERLAP = DIR / "panel-overlap.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r16-precommit.json"
CV_REPORT = REPORT / "r16-cv.json"
RESULT = REPORT / "round16.json"
SEED = 20261216
KEEP = 0.60  # arm p's keep before the veto
KEEP_P = 0.40  # comparator: arm p alone at about the yield arm pv is expected to reach
GOOD_KEPT = 0.80  # veto cut-off: the 80th percentile of max_drop among Good technical-window rows
REPEATS = 10
PER_SOURCE = 30
ARMS = ("b", "s", "gs", "p", "p40", "pv")
TECH = ("r11", "r12", "r13")
SOURCES = ("evals/learned_words_r16.py", "evals/learned_words_r15.py", "evals/learned_words_r14.py", "evals/learned_words_r13.py",
           "models/src/cleartext/learned_words/context_fit.py", "evals/learned_words_corpus.py") + P12.SOURCES


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


doc_ngrams, max_drop = P15.doc_ngrams, P15.max_drop


def cutoff(rows):
    """Round 15's rule: GOOD_KEPT quantile of max_drop on Good technical-window rows."""
    d = np.array([r["drop"] for r in rows if r["tech"] and r["good"]])
    return float(np.quantile(d, GOOD_KEPT))


def pool(with_r13=True):
    """Round 15's 620-row pool (rounds 3-8, 11, 12; veto feature attached) plus round 13's rated
    swaps, whose veto feature removes their source document's counts."""
    rows = P15.pool()
    if with_r13:
        new = P14._rated(P13.RATING, P13.APPLY, "r13")
        key = {"r13:" + s["key"]: s for s in json.load(open(P13.RATING / "key.json"))["swaps"]}
        pdoc = {p["id"]: (p["domain"], p["source_doc"]) for p in json.load(open(P13.PANEL))}
        docs = doc_ngrams({pdoc[key[r["src"]]["pid"]] for r in new})
        for r in new:
            s = key[r["src"]]
            r["drop"] = max_drop(s["before"], s["word"], s["rep"], s.get("start"), docs[pdoc[s["pid"]]])
            r["key"] = list(P12.pair_key(r["word"], r["rep"], r["pos"]))
            r["good"] = int(r["mark"] == "good")
        rows += new
    for r in rows:
        r["tech"] = r["round"] in TECH
    return rows


def _fit(rows, ref, groups):
    """Round 15's final-fit rule: arm p refitted, thresholds at 60% and 40% of the reference rows'
    passage-grouped out-of-fold scores, veto cut-off from Good technical rows."""
    full = np.mean([P14.oof(P12.ArmP, rows, groups, P12.SEED + k) for k in range(REPEATS)], axis=0)
    return P12.ArmP(rows), {"pv": float(np.quantile(full[ref], 1 - KEEP)),
                            "p40": float(np.quantile(full[ref], 1 - KEEP_P)), "cutoff": cutoff(rows)}


def train(argv):
    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    DIR.mkdir(exist_ok=True)
    rows = pool()
    marks = np.array([r["mark"] for r in rows])
    b = np.array([r["b"] for r in rows])
    rnd = np.array([r["round"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    r13 = rnd == "r13"
    ref13 = b & r13
    if (~r13).sum() != 620 or not ref13.any():
        raise SystemExit(f"REFUSED: {(~r13).sum()} earlier rows, {ref13.sum()} round-13 arm-b rows")
    json.dump(rows, open(POOL, "w"))

    # Gate: round 15's arm pv exactly as specified (fit on rounds 3-12), scored on round 13
    early = ~r13
    e_rows = [r for r, x in zip(rows, early) if x]
    ref_e = (b & np.isin(rnd, ("r11", "r12")))[early]
    m15, thr15 = _fit(e_rows, ref_e, groups[early])
    sc = np.zeros(len(rows))
    sc[r13] = m15.score([r for r, x in zip(rows, r13) if x])
    drop = np.array([r["drop"] for r in rows])
    pv = r13 & (sc >= thr15["pv"]) & (drop <= thr15["cutoff"])
    p40 = r13 & (sc >= thr15["p40"])
    res = {"rows": len(rows), "r13_rows": int(r13.sum()), "r13_arm_b_rows": int(ref13.sum()),
           "round15_thresholds": thr15,
           "gate_r13": {"pv": P12._stats(pv, marks, ref13), "p40": P12._stats(p40, marks, ref13),
                        "base": P12._stats(r13, marks, ref13)}}
    g = res["gate_r13"]
    res["go_conditions"] = {"pv_good_ge_0.90": g["pv"]["good_rate"] >= 0.90,
                            "pv_gain_vs_p40_ge_0.03": g["pv"]["good_rate"] >= g["p40"]["good_rate"] + 0.03,
                            "pv_keep_ge_0.35": g["pv"]["keep_share"] >= 0.35}
    res["go"] = bool(all(res["go_conditions"].values()))
    res["pool_sha256"] = _sha(POOL)
    res["source_sha256"] = {p: _sha(p) for p in SOURCES}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "source_sha256"}, indent=1))
    if res["go"]:
        model, thr = _fit(rows, b & np.isin(rnd, TECH), groups)
        pickle.dump(model, open(MODEL, "wb"))
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
        raise SystemExit("REFUSED: models, counts or sense table differ from the round-16 pre-commitment")
    return pickle.load(open(MODEL, "rb")), pickle.load(open(P12.MODEL, "rb")), pre["thresholds"]


# --- fresh window ----------------------------------------------------------------------

def panel():
    """Round 15's panel rule (30 passages per source) with a new seed; excludes every document
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
            out.append({"id": f"r16-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": d,
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
    rng = random.Random(1216)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r16") for x in items] + [dict(x, source="old", old_mark=x["mark"])
                                                     for x in old[:P12.CONSISTENCY_ITEMS]]
    RATING.mkdir(exist_ok=True)
    for r in range(1, P12.RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r16" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    for r in range(1, P12.RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1216:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r16_items": len(items), "items": len(base),
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
            got = [m for m in mp if m["source"] == "r16" and arm in m["arms"]]
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
    r16 = [x["id"] for x in mp if x["source"] == "r16"]
    out["unanimous_share"] = sum(len({rt[i] for rt in raters}) == 1 for i in r16) / max(1, len(r16))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r16) / max(1, len(r16))
                                 for a in range(P12.RATERS) for c in range(a + 1, P12.RATERS)}
    old = [x for x in mp if x["source"] == "old"]
    out["consistency_with_old"] = [sum(x["old_mark"] == rt[x["id"]] for x in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"train": train, "apply": apply, "page": page, "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r16 train|apply|page|score")
