"""Round 23: round 22's arms frozen on a new genre window, a replication (see ROUND23.md).

  learned-words r23 precommit                fix round 22's models and thresholds (no model is fitted)
  learned-words r23 apply [--max-seconds N]  fresh news + encyclopedic panel (once), round 11's generators (checkpointed)
  learned-words r23 page | score             three blind rater files; majority-vote scoring

Raw text and marks stay in the state directory (r23/).
"""

import csv
import hashlib
import json
import pickle
import random
import time
from collections import Counter

import numpy as np

from . import learned_words_genre as GEN
from . import learned_words_r12 as P12
from . import learned_words_r14 as P14
from . import learned_words_r19 as P19
from . import learned_words_r21 as Q
from . import learned_words_r22 as Q22
from . import learned_words_r9 as R
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r23"
POOL = Q22.POOL
MODEL_QG = Q22.MODEL_QG
MODEL_RG = Q22.MODEL_RG
PANEL = DIR / "passages.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r23-precommit.json"
CV_REPORT = REPORT / "r23-cv.json"
RESULT = REPORT / "round23.json"
SEED = 20261223
PER_SOURCE = {"news": 45, "wiki": 60}
KEEP_GENRE = 0.40  # qg, rg: share of genre arm-b rows (round 21's news + encyclopedic) kept
KEEP_TECH = 0.50  # the technical threshold, as round 21's (reported, not used on this window)
REPEATS = 10
POOL_ROWS = 1259 + 118
ARMS = ("b", "gs", "p50", "q50", "rg", "qg")
SOURCES = ("evals/learned_words_r23.py", "evals/learned_words_r22.py", "evals/learned_words_joshua.py") + Q.SOURCES


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def precommit(argv):
    """Round 23 fits nothing: it fixes round 22's refitted qg and rg, their genre thresholds and the
    frozen p50 and q50, after checking them against round 22's pre-commitment."""
    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    pre22 = json.load(open(Q22.PRECOMMIT))
    gate = {"r22_check_go": bool(json.load(open(Q22.CV_REPORT))["go"]),
            "qg_model_unchanged": _sha(MODEL_QG) == pre22["model_qg_sha256"],
            "rg_model_unchanged": _sha(MODEL_RG) == pre22["model_rg_sha256"],
            "q50_model_unchanged": _sha(Q.MODEL_Q) == pre22["model_q50_sha256"],
            "p50_model_unchanged": _sha(P19.MODEL) == pre22["model_p50_sha256"]}
    print(json.dumps(gate, indent=1))
    if not all(gate.values()):
        raise SystemExit("NO-GO: no fresh text is spent")
    DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"gate": gate, "thresholds": pre22["thresholds"], "r22_precommit_sha256": _sha(Q22.PRECOMMIT),
               "model_qg_sha256": pre22["model_qg_sha256"], "model_rg_sha256": pre22["model_rg_sha256"],
               "model_q50_sha256": pre22["model_q50_sha256"], "model_p50_sha256": pre22["model_p50_sha256"],
               "sense_table_sha256": _sha(R.TABLE), "arm_s_model_sha256": _sha(S.MODEL),
               "genre_corpus_sha256": {x: _sha(GEN.GENRE / f"{x}.jsonl.gz") for x in Q.GENRES},
               "source_sha256": {p: _sha(p) for p in SOURCES}, "panel_seed": SEED, "per_source": PER_SOURCE,
               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
    print(open(PRECOMMIT).read())


def load_models():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL_QG) != pre["model_qg_sha256"] or _sha(MODEL_RG) != pre["model_rg_sha256"] \
            or _sha(Q.MODEL_Q) != pre["model_q50_sha256"] or _sha(P19.MODEL) != pre["model_p50_sha256"] \
            or _sha(R.TABLE) != pre["sense_table_sha256"] or _sha(S.MODEL) != pre["arm_s_model_sha256"]:
        raise SystemExit("REFUSED: models or sense table differ from the round-22 pre-commitment")
    return {"qg": pickle.load(open(MODEL_QG, "rb")), "rg": pickle.load(open(MODEL_RG, "rb")),
            "q50": pickle.load(open(Q.MODEL_Q, "rb")), "p50": pickle.load(open(P19.MODEL, "rb"))}, pre


def panel():
    """Round 22's panel rule (seed SEED), also excluding every document of round 22's window and Joshua's
    supplement, and any passage sharing a word 8-gram with them."""
    from cleartext.learned_words.rewrite import tokens

    if PANEL.exists():
        raise SystemExit(f"REFUSED: {PANEL} exists")
    prev = []
    for p in (R.PANEL, Q.P12.PANEL, Q.P13.PANEL, Q.P17.PANEL, Q.P18.PANEL, Q.P19.PANEL, Q.P20.PANEL,
              Q.DIR / "passages.json", Q.JDIR / "passages.json", Q22.PANEL, STATE / "joshua-r21b/passages.json"):
        prev += json.load(open(p))
    used = [x["original"] for x in prev]
    for p in ("panel-r4", "panel-r5", "panel-r8"):
        used += [x["original"] for x in json.load(open(STATE / p / "passages.json"))]
    used += [s["before"] for s in R.dev_swaps()]
    used += [x["original"] for x in json.load(open(Q.V3_PANEL))]
    grams = P12._grams(used)
    prev_docs = {(x["domain"], x["source_doc"]) for x in prev}
    out = []
    for src, want in PER_SOURCE.items():
        docs = Q._docs(src)
        order = sorted(docs)
        rng = random.Random(f"{SEED}:{src}")
        rng.shuffle(order)
        n = 0
        for doc in order:
            if n >= want:
                break
            if (src, doc) in prev_docs:
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
            out.append({"id": f"r23-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": doc,
                        "first_sentence": i, "original": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(PANEL, "w"), indent=1)
    print(json.dumps({"passages": len(out), "by_source": Counter(p["domain"] for p in out)}), flush=True)


def _overlap(pan):
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
    for _, _, a, c in pairs():
        for x in (a, c):
            t = tokens(x)
            for i in range(len(t) - 7):
                hit |= index.get(tuple(t[i:i + 8]), set())
    json.dump(sorted(hit), open(path, "w"))
    return hit


def apply(argv):
    """Round 21's apply on the round-22 window (checkpointed)."""
    from cleartext.learned_words import senses as SN

    from .learned_splits import sent
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    R._threads(argv)
    load_models()
    deadline = time.monotonic() + R._seconds(argv, 1200)
    if not PANEL.exists():
        panel()
    pan = json.load(open(PANEL))
    skip = _overlap(pan)
    done = {r["pid"] for r in read_jsonl(APPLY)}
    todo = [p for p in pan if p["id"] not in done and p["id"] not in skip]
    print(f"{len(done)} done, {len(todo)} to do, {len(skip)} skipped for rewrite-corpus overlap", flush=True)
    if todo:
        clf_s, _ = S.load_filter()
        sn = SN.load(R.TABLE)
        _, T, ids, _ = load()
        model = pickle.load(open(STATE / "model.pkl", "rb"))
        simpler = pickle.load(open(SIMPLER, "rb"))
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
                    X = Q.P13._x31(T, ids, got, rows)
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


def picks(rec, models, thr, choose):
    """b, gs as before; p50 (round 19) and q50 (round 21) frozen at their thresholds; rg and qg refitted
    on 1,377 rows at the genre threshold. Filters apply before the per-passage limits."""
    a = [s for s in rec["all"] if s["margin"] >= S.MIN_MARGIN]
    g = [s for s in rec["sense"] if s["margin"] >= S.MIN_MARGIN]
    if a:
        rq = Q._rows_q(a)
        for name in ("p50", "rg"):
            for s, x in zip(a, P12.score_p(models[name], a)):
                s["sc_" + name] = float(x)
        for name in ("q50", "qg"):
            for s, x in zip(a, models[name].score(rq)):
                s["sc_" + name] = float(x)
    out = {"b": choose(a), "gs": choose([s for s in g if s["stack"] >= thr["s"]])}
    for name in ("p50", "q50", "rg", "qg"):
        out[name] = choose([s for s in a if s["sc_" + name] >= thr[name]])
    return out


def page(argv):
    from .learned_words_apply import _skey, choose

    models, pre = load_models()
    items = {}
    for r in read_jsonl(APPLY):
        for arm, chosen in picks(r, models, pre["thresholds"], choose).items():
            for s in chosen:
                k = _skey(s)
                it = items.setdefault(k, dict({x: v for x, v in s.items() if x != "x31"}, key=k, arms=[],
                                              domain=r["domain"], shown=Q.realized(s)))
                if arm not in it["arms"]:
                    it["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1223)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r23") for x in items] + [dict(x, source="old", old_mark=x["mark"], shown=x["after"])
                                                     for x in old[:P12.CONSISTENCY_ITEMS]]
    RATING.mkdir(parents=True, exist_ok=True)
    for r in range(1, P12.RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r23" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    for r in range(1, P12.RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1223:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["shown"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r23_items": len(items), "items": len(base), "realized": sum(x["shown"] != x["after"] for x in items),
                      "per_arm": Counter(a for x in items for a in x["arms"]),
                      "per_domain": Counter(x["domain"] for x in items)}, indent=1))


def score(argv):
    from .learned_words_study import wilson

    load_models()
    pool_pairs = {tuple(r["key"]) for r in json.load(open(POOL))}
    pool21 = {tuple(r["key"]) for r in Q.pool()}
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
            got = [m for m in mp if m["source"] == "r23" and arm in m["arms"]]
            ms = [marks[m["id"]] for m in got]
            c = Counter(ms)
            un = [marks[m["id"]] for m in got if pk(m) not in pool_pairs]
            un21 = [marks[m["id"]] for m in got if pk(m) not in pool21]
            res["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(c["good"], len(ms)), "meaning_rate": c["meaning"] / max(1, len(ms)),
                                 "per_100_sentences": 100 * len(ms) / max(1, sentences),
                                 "distinct_pairs": len({pk(m) for m in got}),
                                 "unseen_pairs": {"swaps": len(un), "good": un.count("good")},
                                 "unseen_before_r21": {"swaps": len(un21), "good": un21.count("good")},
                                 "by_domain": {d: {"swaps": sum(key[m["key"]]["domain"] == d for m in got),
                                                   "good": sum(key[m["key"]]["domain"] == d and marks[m["id"]] == "good"
                                                               for m in got)} for d in Q.GENRES}}
        return res

    out = {"items": len(ids), "raters": P12.RATERS, "passages": len(recs), "sentences": sentences,
           "majority": arm_stats(maj), "per_rater": [arm_stats(m) for m in raters]}
    M = out["majority"]
    out["yield_vs_b"] = {a: M["arm_" + a]["swaps"] / max(1, M["arm_b"]["swaps"]) for a in ARMS}
    v = M["arm_qg"]
    out["success_qg"] = {"good_ge_0.90": v["good_rate"] >= 0.90, "meaning_le_0.08": v["meaning_rate"] <= 0.08,
                         "yield_ge_0.40": out["yield_vs_b"]["qg"] >= 0.40}
    out["success"] = bool(all(out["success_qg"].values()))
    r23 = [x["id"] for x in mp if x["source"] == "r23"]
    out["unanimous_share"] = sum(len({rt[i] for rt in raters}) == 1 for i in r23) / max(1, len(r23))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r23) / max(1, len(r23))
                                 for a in range(P12.RATERS) for c in range(a + 1, P12.RATERS)}
    old = [x for x in mp if x["source"] == "old"]
    out["consistency_with_old"] = [sum(x["old_mark"] == rt[x["id"]] for x in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"precommit": precommit, "apply": apply, "page": page, "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r23 precommit|apply|page|score")
