"""Round 13: the frozen round-12 arm p on a second fresh window, plus a runner-up fallback
(see ROUND13.md).

  learned-words r13 smoke                          3-passage dry run on spent round-11 text
  learned-words r13 precommit                      gate check and pre-commitment (no model is fitted)
  learned-words r13 apply [--max-seconds N]        fresh panel (once) and three generators (checkpointed)
  learned-words r13 page | score                   three blind rater files; majority-vote scoring
  learned-words r13 features11 [--max-seconds N]   the 31 arm-s features of round 11's rated swaps (for later rounds)

Raw text, features and marks stay in the state directory (r13/).
"""

import gzip
import hashlib
import json
import pickle
import random
import time
from collections import Counter

import numpy as np

from . import learned_words_r12 as P12
from . import learned_words_r9 as R
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r13"
FEATURES11 = DIR / "features11.jsonl"
PANEL = DIR / "passages.json"
OVERLAP = DIR / "panel-overlap.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r13-precommit.json"
RESULT = REPORT / "round13.json"
SEED = 20261213
PER_SOURCE = 45
PASSAGE_SENTENCES = 4


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def panel():
    """Round 12's panel rule with a new seed; also excludes every round-12 document and any word
    8-gram shared with the round-12 panel."""
    from cleartext.learned_words.rewrite import tokens

    if PANEL.exists():
        raise SystemExit(f"REFUSED: {PANEL} exists")
    prev = json.load(open(R.PANEL)) + json.load(open(P12.PANEL))
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
            starts = [i for i in range(len(sents) - PASSAGE_SENTENCES + 1)
                      if all(R._ok_sentence(x) for x in sents[i:i + PASSAGE_SENTENCES])]
            if not starts:
                continue
            i = rng.choice(starts)
            text = " ".join(sents[i:i + PASSAGE_SENTENCES])
            tt = tokens(text)
            if any(tuple(tt[k:k + 8]) in grams for k in range(len(tt) - 7)):
                continue
            n += 1
            out.append({"id": f"r13-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": d,
                        "first_sentence": i, "original": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    json.dump(out, open(PANEL, "w"), indent=1)
    print(json.dumps({"passages": len(out), "by_source": Counter(p["domain"] for p in out)}, indent=1), flush=True)


def _overlap(pan):
    """Passage ids sharing a word 8-gram with the round-7 rewrite corpora (cached)."""
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


def _x31(T, ids, swaps, rows):
    A = np.array([[r["f"][k] for k in S.STAGE_KEYS] for r in rows], float)
    spans = [dict(s, start=r["start"]) for s, r in zip(swaps, rows)]
    return np.nan_to_num(np.hstack([A, S.rewrite_features(T, ids, spans)]))


def features11(argv):
    """Re-featurize the 135 round-11 rated swaps (stage path of round 11), checkpointed. The
    committed arm-s model must reproduce each stored arm-s score from these features."""
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    R._threads(argv)
    DIR.mkdir(exist_ok=True)
    deadline = time.monotonic() + R._seconds(argv, 1380)
    clf, _ = S.load_filter()
    _, T, ids, _ = load()
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    simpler = pickle.load(open(SIMPLER, "rb"))
    swaps = json.load(open(R.RATING11 / "key.json"))["swaps"]
    done = {r["key"] for r in read_jsonl(FEATURES11)}
    with open(FEATURES11, "a") as f:
        for s in swaps:
            if s["key"] in done:
                continue
            if time.monotonic() > deadline:
                break
            r = S.swap_row(model, simpler, s)
            rec = {"key": s["key"], "x31": None}
            if r is not None:
                x = _x31(T, ids, [s], [r])
                rec.update(x31=[float(v) for v in x[0]], lemma=r["lemma"], start=r["start"],
                           stack=float(clf.predict_proba(x)[0, 1]), stack_stored=float(s["stack"]))
            f.write(json.dumps(rec) + "\n")
            f.flush()
    rows = read_jsonl(FEATURES11)
    got = [r for r in rows if r["x31"] is not None]
    diff = max((abs(r["stack"] - r["stack_stored"]) for r in got), default=0.0)
    print(f"{len(rows)} of {len(swaps)} rows, {len(rows) - len(got)} unresolved, max |stack - stored| {diff:.2e}",
          flush=True)
    print("COMPLETE" if len(rows) == len(swaps) else "PARTIAL", flush=True)


def word_swaps_three(text, model, pid, sn):
    """Round 11's two generators plus a third from the same featurization per target.
    all: every candidate (arm b's generator). sense: candidates covering >= 0.25 of the target's
    in-context sense mass (arm g's). next: every candidate except the all-variant's top one, so
    the stage decides again on its runner-up. Mirrors R.word_swaps_two."""
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
    out = {"all": [], "next": [], "sense": []}
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
        cover = R.in_sense(sn, s, inst["start"], inst["end"], rows[0]["lemma"], rows[0]["pos"],
                           [r["word"] for r in rows])
        cov = {id(r): c for r, c in zip(rows, cover)}
        top_all = None
        for name in ("all", "next", "sense"):
            if name == "all":
                rs = rows
            elif name == "next":
                rs = [r for r in rows if r is not top_all]
            else:
                rs = [r for r in rows if cov[id(r)] >= R.SENSE_COVER]
            if not rs:
                continue
            d, top = R._decide_top(model, F, decide, inst, rs, s)
            if name == "all":
                top_all = top
            if d is None or not d["swap"]:
                continue
            after = s[:inst["start"]] + d["word"] + s[inst["end"]:]
            out[name].append({"pid": pid, "sentence": si, "before": s, "after": after, "word": t.text,
                              "rep": d["word"], "pos": d["pos"], "gate": d["gate"], "check": d["check"],
                              "p_complex": pc, "p_term": pt, "sources": top["sources"], "cover": cov[id(top)],
                              "target_id": iid, "variant": name})
    return out, seen


def smoke(argv):
    """Dry run on 3 round-11 passages (old, spent text): the all and sense variants must equal
    the stored round-11 swaps of those passages; prints the next-variant swaps' count only."""
    from cleartext.learned_words import senses as SN

    R._threads(argv)
    sn = SN.load(R.TABLE)
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    pan = {p["id"]: p for p in json.load(open(R.PANEL))}
    old = {r["pid"]: r for r in read_jsonl(R.APPLY11)}
    ok = True
    for pid in [p for p in old if old[p]["all"]][:3]:
        v, _ = word_swaps_three(pan[pid]["original"], model, pid, sn)
        for name in ("all", "sense"):
            a = [(x["word"], x["rep"], x["sentence"]) for x in v[name]]
            b = [(x["word"], x["rep"], x["sentence"]) for x in old[pid][name]]
            ok &= a == b
            print(pid, name, "match" if a == b else f"DIFFER {a} vs {b}", flush=True)
        print(pid, "next", len(v["next"]), flush=True)
    print("COMPLETE" if ok else "FAILED", flush=True)


def apply(argv):
    """The three generators per target, the arm-s score and the 31 arm-s features of every
    resulting swap (checkpointed)."""
    from cleartext.learned_words import senses as SN

    from .learned_splits import sent
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    R._threads(argv)
    load_p13()
    DIR.mkdir(exist_ok=True)
    if not PANEL.exists():
        panel()
    deadline = time.monotonic() + R._seconds(argv, 1380)
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
            variants, seen = word_swaps_three(p["original"], model, p["id"], sn)
            rec = {"pid": p["id"], "domain": p["domain"], "content_words": seen, "sentences": len(sent(p["original"]))}
            for name, swaps in variants.items():
                got, rows = [], []
                for s in swaps:
                    r = S.swap_row(model, simpler, s)
                    if r is not None:
                        got.append(s)
                        rows.append(r)
                if got:
                    X = _x31(T, ids, got, rows)
                    for s, r, x, sc in zip(got, rows, X, clf_s.predict_proba(X)[:, 1]):
                        s.update(margin=r["f"]["margin"], stack=float(sc), x31=[float(v) for v in x],
                                 lemma=r["lemma"], start=r["start"])
                rec[name] = [s for s in swaps if "stack" in s]
                rec[f"unresolved_{name}"] = len(swaps) - len(rec[name])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} " + " ".join(f"{k}={len(rec[k])}" for k in variants), flush=True)
    left = len({p["id"] for p in pan} - skip - {r["pid"] for r in read_jsonl(APPLY)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


# --- pre-commitment, gate and fresh window ---------------------------------------------

ARMS = ("b", "s", "gs", "t", "p", "p65", "pn", "p65n")
SOURCES = ("evals/learned_words_r13.py",) + P12.SOURCES
SMOKE_LOG = DIR / "smoke.log"


def precommit(argv):
    """Round 13 changes no model: it fixes the frozen round-12 models and thresholds and the new
    generator's source, after the round-12 gate and the smoke check, before any round-13 swap."""
    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    cv = json.load(open(P12.CV_REPORT))
    pre12 = json.load(open(P12.PRECOMMIT))
    smoke_ok = SMOKE_LOG.exists() and SMOKE_LOG.read_text().strip().endswith("COMPLETE")
    gate = {"r12_cv_go": bool(cv["go"]), "r12_model_unchanged": _sha(P12.MODEL) == pre12["model_sha256"],
            "smoke_reproduces_round11": smoke_ok}
    print(json.dumps(gate, indent=1))
    if not all(gate.values()):
        raise SystemExit("NO-GO: no fresh text is spent")
    json.dump({"gate": gate, "thresholds": pre12["thresholds"], "r12_model_sha256": pre12["model_sha256"],
               "r12_precommit_sha256": _sha(P12.PRECOMMIT), "sense_table_sha256": _sha(R.TABLE),
               "arm_s_model_sha256": _sha(S.MODEL), "smoke_log_sha256": _sha(SMOKE_LOG),
               "source_sha256": {p: _sha(p) for p in SOURCES}, "panel_seed": SEED,
               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
    print("COMPLETE")


def load_p13():
    pre = json.load(open(PRECOMMIT))
    if _sha(P12.MODEL) != pre["r12_model_sha256"] or _sha(R.TABLE) != pre["sense_table_sha256"] \
            or _sha(S.MODEL) != pre["arm_s_model_sha256"]:
        raise SystemExit("REFUSED: models or sense table differ from the round-13 pre-commitment")
    return pickle.load(open(P12.MODEL, "rb")), pre["thresholds"]


def fallback(primary, runner_up, ok):
    """Primary swaps that pass `ok`; for every other target, its runner-up swap if that passes."""
    out = [s for s in primary if ok(s)]
    used = {s["target_id"] for s in out}
    return out + [s for s in runner_up if s["target_id"] not in used and ok(s)]


def picks(rec, model, thr, choose):
    """Every arm's swaps in one passage; filters (and the fallback) apply before the per-passage
    limits. b, s, gs, t, p, p65 as in round 12. pn, p65n: arm p (p65) on each target's top
    swap, else on its runner-up swap."""
    a = [s for s in rec["all"] if s["margin"] >= S.MIN_MARGIN]
    n = [s for s in rec["next"] if s["margin"] >= S.MIN_MARGIN]
    g = [s for s in rec["sense"] if s["margin"] >= S.MIN_MARGIN]
    for xs in (a, n):
        for s, x in zip(xs, P12.score_p(model, xs)):
            s["p"] = float(x)
    return {"b": choose(a), "s": choose([s for s in a if s["stack"] >= thr["s"]]),
            "gs": choose([s for s in g if s["stack"] >= thr["s"]]),
            "t": choose([s for s in a if s["stack"] >= thr["t"]]),
            "p": choose([s for s in a if s["p"] >= thr["p"]]),
            "p65": choose([s for s in a if s["p"] >= thr["p65"]]),
            "pn": choose(fallback(a, n, lambda s: s["p"] >= thr["p"])),
            "p65n": choose(fallback(a, n, lambda s: s["p"] >= thr["p65"]))}


def page(argv):
    from .learned_words_apply import _skey, choose

    model, thr = load_p13()
    items = {}
    for r in read_jsonl(APPLY):
        for arm, chosen in picks(r, model, thr, choose).items():
            for s in chosen:
                k = _skey(s)
                it = items.setdefault(k, dict({x: v for x, v in s.items() if x != "x31"}, key=k, arms=[],
                                              domain=r["domain"]))
                if arm not in it["arms"]:
                    it["arms"].append(arm)
                if s.get("variant") == "next":
                    it["runner_up"] = True
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1213)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r13") for x in items] + [dict(x, source="old", old_mark=x["mark"])
                                                     for x in old[:P12.CONSISTENCY_ITEMS]]
    RATING.mkdir(exist_ok=True)
    for r in range(1, P12.RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r13" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    for r in range(1, P12.RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1213:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r13_items": len(items), "items": len(base),
                      "per_arm": Counter(a for x in items for a in x["arms"])}, indent=1))


def score(argv):
    import csv

    from .learned_words_study import wilson

    model, _ = load_p13()
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
            got = [m for m in mp if m["source"] == "r13" and arm in m["arms"]]
            ms = [marks[m["id"]] for m in got]
            c = Counter(ms)
            unseen = [marks[m["id"]] for m in got if not model.pri.seen(pk(m))]
            runner = [marks[m["id"]] for m in got if key[m["key"]].get("runner_up")]
            res["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(c["good"], len(ms)), "meaning_rate": c["meaning"] / max(1, len(ms)),
                                 "per_100_sentences": 100 * len(ms) / max(1, sentences),
                                 "distinct_pairs": len({pk(m) for m in got}),
                                 "unseen_pairs": {"swaps": len(unseen), "good": unseen.count("good")},
                                 "runner_up_swaps": {"swaps": len(runner), "good": runner.count("good")}}
        return res

    out = {"items": len(ids), "raters": P12.RATERS, "passages": len(recs), "sentences": sentences,
           "majority": arm_stats(maj), "per_rater": [arm_stats(m) for m in raters]}
    M = out["majority"]
    out["yield_vs_b"] = {a: M["arm_" + a]["swaps"] / max(1, M["arm_b"]["swaps"]) for a in ARMS}
    pn = M["arm_pn"]
    out["success"] = bool(pn["good_rate"] >= 0.90 and pn["meaning_rate"] <= 0.08 and out["yield_vs_b"]["pn"] >= 0.40
                          and pn["swaps"] > M["arm_p"]["swaps"])
    out["replication_p"] = bool(M["arm_p"]["good_rate"] >= 0.90 and M["arm_p"]["meaning_rate"] <= 0.08
                                and out["yield_vs_b"]["p"] >= 0.40)
    r13 = [m["id"] for m in mp if m["source"] == "r13"]
    out["unanimous_share"] = sum(len({m[i] for m in raters}) == 1 for i in r13) / max(1, len(r13))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r13) / max(1, len(r13))
                                 for a in range(P12.RATERS) for c in range(a + 1, P12.RATERS)}
    old = [m for m in mp if m["source"] == "old"]
    out["consistency_with_old"] = [sum(m["old_mark"] == rt[m["id"]] for m in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"features11": features11, "smoke": smoke, "precommit": precommit, "apply": apply, "page": page,
           "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r13 features11|smoke|precommit|apply|page|score")
