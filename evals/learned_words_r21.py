"""Round 21: unseen pairs, meaning residue and a genre test of frozen p50 (see ROUND21.md).

  learned-words r21 unseen                   unseen-pair CV report of arm p (r21-unseen.json); fits nothing kept
  learned-words r21 train                    CV gate for arm q50; final fits and pre-commit if go
  learned-words r21 apply [--max-seconds N]  fresh panels (once) and round 11's two generators (checkpointed):
                                             the rated round-21 window first, then Joshua's window
  learned-words r21 page | score             three blind rater files; majority-vote scoring
  learned-words r21 joshua-page              Joshua's blind marking page (never rated by an AI rater)

Raw text, pools, keys and marks stay in the state directory (r21/, joshua-r21/).
"""

import csv
import gzip
import hashlib
import html
import json
import pickle
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import learned_words_genre as GEN
from . import learned_words_r12 as P12
from . import learned_words_r13 as P13
from . import learned_words_r14 as P14
from . import learned_words_r17 as P17
from . import learned_words_r18 as P18
from . import learned_words_r19 as P19
from . import learned_words_r20 as P20
from . import learned_words_r9 as R
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r21"
POOL = DIR / "pool.json"
MODEL_Q = DIR / "model-q.pkl"
MODEL_R = DIR / "model-r.pkl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r21-precommit.json"
CV_REPORT = REPORT / "r21-cv.json"
UNSEEN_REPORT = REPORT / "r21-unseen.json"
RESULT = REPORT / "round21.json"
JDIR = STATE / "joshua-r21"
JPAGE = Path.home() / "Documents/cleartext-rating/rate-learned-swaps-r21.html"
V3_PANEL = Path.home() / ".local/state/cleartext/testset-v3-20261116/panel/all.json"  # exclusion only
WINDOWS = {  # name: (directory, seed, passages per source)
    "r21": (DIR, 20261221, {"news": 45, "wiki": 45, "fedreg": 23, "openfda": 22}),
    "joshua": (JDIR, 20261231, {"news": 45, "wiki": 45, "fedreg": 30, "openfda": 30}),
}
GENRES = ("news", "wiki")
KEEP = 0.50
REPEATS = 10
TECH = ("r11", "r12", "r13", "r17", "r18", "r19", "r20")
ARMS = ("b", "s", "gs", "p", "p50", "r50", "q50")
POOL_ROWS = 1259
J_BEST, J_B = 60, 40  # Joshua's page: swaps from the best arm, and arm-b swaps that arm rejects
SOURCES = ("evals/learned_words_r21.py", "evals/learned_words_genre.py", "models/src/cleartext/learned_words/realize.py",
           "evals/learned_words_r20.py") + P19.SOURCES


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# --- features ------------------------------------------------------------------------

def _stem():
    from nltk.stem import SnowballStemmer

    return SnowballStemmer("english").stem


STEM = None
_TOK = re.compile(r"[A-Za-z]+")
SK = list(S.STAGE_KEYS)


def start_of(before, after):
    """Character offset of the swapped word in `before` (first difference, back to a word start)."""
    i = 0
    while i < min(len(after), len(before)) and after[i] == before[i]:
        i += 1
    while i > 0 and before[i - 1].isalnum():
        i -= 1
    return i


def context(before, after, rep):
    """Two hand-made context features (learned weights, never a veto): the target is inside
    quotation marks; the replacement's stem already occurs elsewhere in the sentence."""
    global STEM
    STEM = STEM or _stem()
    i = start_of(before, after)
    pre = before[:i]
    quoted = (pre.count('"') % 2 == 1) or (pre.count("“") > pre.count("”"))
    stems = {STEM(m.group().lower()) for m in _TOK.finditer(before) if not (m.start() <= i < m.end())}
    return [float(quoted), float(STEM(rep.lower()) in stems)]


class _Rates:
    def __init__(self, keys, good, a=P12.SMOOTH):
        self.a, self.m, self.c = a, float(np.mean(good)), {}
        for k, g in zip(keys, good):
            c = self.c.setdefault(k, [0, 0])
            c[0] += int(g)
            c[1] += 1

    def rate(self, k, own=None):
        g, n = self.c.get(k, (0, 0))
        if own is not None:
            g, n = g - int(own), n - 1
        return (g + self.a * self.m) / (n + self.a), n


class ArmQ:
    """Arm q: arm p's four features plus back-off priors (replacement lemma, stem-family pair,
    target stem family), the pair's count and an unseen-pair x arm-s interaction (so unseen
    pairs can lean on the arm-s score), two context features, and two stage meaning signals
    (ctx_cos, checker score). Standardized logistic, C = 0.1; priors leave the row out."""

    def __init__(self, rows):
        global STEM
        STEM = STEM or _stem()
        y = [r["good"] for r in rows]
        self.pair = _Rates([tuple(r["key"]) for r in rows], y)
        self.tgt = _Rates([r["key"][0] for r in rows], y)
        self.rep = _Rates([r["key"][1] for r in rows], y)
        self.fam = _Rates([(STEM(r["key"][0]), STEM(r["key"][1])) for r in rows], y)
        self.tfam = _Rates([STEM(r["key"][0]) for r in rows], y)
        self.clf = S._fit(self.design(rows, loo=True), np.array(y))

    def seen(self, key):
        return tuple(key) in self.pair.c

    def design(self, rows, loo):
        global STEM
        STEM = STEM or _stem()
        X = []
        for r in rows:
            own = r["good"] if loo else None
            t, c = r["key"]
            pp, n = self.pair.rate((t, c), own)
            ls = P12._logit(r["stack"])
            X.append([ls, float(r["cover"]), P12._logit(pp), P12._logit(self.tgt.rate(t, own)[0]),
                      P12._logit(self.rep.rate(c, own)[0]), P12._logit(self.fam.rate((STEM(t), STEM(c)), own)[0]),
                      P12._logit(self.tfam.rate(STEM(t), own)[0]), float(np.log1p(max(n, 0))), float(n <= 0) * ls,
                      *r["ctx"], r["x31"][SK.index("ctx_cos")], r["x31"][SK.index("check")]])
        return np.nan_to_num(np.array(X, float))

    def score(self, rows):
        return self.clf.predict_proba(self.design(rows, loo=False))[:, 1]


# --- pool ------------------------------------------------------------------------------

def pool():
    """Round 19's 1,024-row pool plus rounds 19 and 20's rated swaps, with sentence text and the
    two context features."""
    rows = json.load(open(P19.POOL))
    rows += P14._rated(P19.RATING, P19.APPLY, "r19") + P14._rated(P20.RATING, P20.APPLY, "r20")
    items = json.load(open(STATE / "replay/items.json"))
    keys = {}
    for d in [STATE / "rating-r8", R.RATING11] + [STATE / f"r{n}/rating" for n in (12, 13, 17, 18, 19, 20)]:
        for s in json.load(open(d / "key.json"))["swaps"]:
            keys[s["key"]] = s
    for r in rows:
        r["key"] = list(P12.pair_key(r["word"], r["rep"], r["pos"]))
        r["good"] = int(r["mark"] == "good")
        r["tech"] = r["round"] in TECH
        tag, k = r["src"].split(":", 1)
        s = items[int(k)] if tag == "replay" else keys[k]
        r["before"], r["after"], r["domain"] = s["before"], s["after"], s.get("domain")
        r["ctx"] = context(r["before"], r["after"], r["rep"])
    return rows


def _arrays(rows):
    return (np.array([r["mark"] for r in rows]), np.array([r["b"] for r in rows]),
            np.array([r["round"] for r in rows]), np.array([r["group"] for r in rows]),
            np.array(["|".join(r["key"]) for r in rows]), np.array([r["key"][0] for r in rows]))


def nested_seen(cls, rows, groups, ref, seed, keep, keyp):
    """P14.nested plus, per held-out row, whether its pair occurs in the training folds."""
    score, kp, seen = np.zeros(len(rows)), np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    for tr, te in P12._folds(groups, seed):
        sub = [rows[i] for i in tr]
        inner = P14.oof(cls, sub, groups[tr], seed + 1)
        t = np.quantile(inner[ref[tr]], 1 - keep)
        score[te] = cls(sub).score([rows[i] for i in te])
        kp[te] = score[te] >= t
        sp = set(keyp[tr])
        seen[te] = [k in sp for k in keyp[te]]
    return score, kp, seen


def _cv(cls, rows, grouping, repeats, keep=KEEP):
    marks, b, rnd, groups, keyp, keyt = _arrays(rows)
    ref = b & np.isin(rnd, TECH)
    g = {"passage": groups, "pair": keyp, "lemma": keyt}[grouping]
    runs = []
    for k in range(repeats):
        _, kp, seen = nested_seen(cls, rows, g, ref, P12.SEED + 10 * k, keep, keyp)
        runs.append({"all": P12._stats(kp, marks, ref), "seen_pair": P12._stats(kp, marks, ref & seen),
                     "unseen_pair": P12._stats(kp, marks, ref & ~seen),
                     "unseen_pair_rows": int((ref & ~seen).sum())})
    out = {s: {f: float(np.mean([r[s][f] for r in runs])) for f in runs[0][s]} for s in ("all", "seen_pair", "unseen_pair")}
    out["unseen_pair_rows"] = float(np.mean([r["unseen_pair_rows"] for r in runs]))
    out["good_rate_by_repeat"] = [r["all"]["good_rate"] for r in runs]
    return out


def unseen(argv):
    """Honest unseen-pair measurement of arm p (round 19's model class) on all 1,259 rated rows:
    passage-grouped CV split by whether the held-out pair was seen in training folds, and pair- and
    target-lemma-grouped CV where no held-out pair (or target) was seen. Nothing is kept."""
    from sklearn.metrics import roc_auc_score

    DIR.mkdir(exist_ok=True)
    rows = pool()
    if len(rows) != POOL_ROWS:
        raise SystemExit(f"REFUSED: pool has {len(rows)} rows")
    marks, b, rnd, groups, keyp, keyt = _arrays(rows)
    ref = b & np.isin(rnd, TECH)
    res = {"rows": len(rows), "ref_rows": int(ref.sum()), "keep": KEEP, "repeats": 5}
    for name, cls in (("p", P12.ArmP), ("q", ArmQ)):
        res[name] = {g: _cv(cls, rows, g, 5) for g in ("passage", "pair", "lemma")}
    # signal check inside passage-CV unseen-pair rows: AUC and precision of the top share by each score
    y = marks == "good"
    sig = {}
    for k in range(5):
        seed = P12.SEED + 10 * k
        sc = {n: np.zeros(len(rows)) for n in ("p", "q")}
        seen = np.zeros(len(rows), bool)
        for tr, te in P12._folds(groups, seed):
            sub = [rows[i] for i in tr]
            for n, c in (("p", P12.ArmP), ("q", ArmQ)):
                sc[n][te] = c(sub).score([rows[i] for i in te])
            sp = set(keyp[tr])
            seen[te] = [x in sp for x in keyp[te]]
        sc["arm_s"] = np.array([r["stack"] for r in rows])
        on = ref & ~seen
        for n, s in sc.items():
            o = np.argsort(-s[on])
            yy = y[on][o]
            sig.setdefault(n, []).append({"auc": float(roc_auc_score(y[on], s[on])), "base_good": float(y[on].mean()),
                                          **{f"good_top_{int(f * 100)}pct": float(yy[:max(1, int(f * len(yy)))].mean())
                                             for f in (0.1, 0.2, 0.3, 0.5)}})
    res["unseen_rows_signal"] = {n: {f: float(np.mean([x[f] for x in v])) for f in v[0]} for n, v in sig.items()}
    res["source_sha256"] = {p: _sha(p) for p in SOURCES}
    json.dump(res, open(UNSEEN_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "source_sha256"}, indent=1))


# --- CV gate and pre-commitment --------------------------------------------------------

def train(argv):
    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    DIR.mkdir(exist_ok=True)
    rows = pool()
    if len(rows) != POOL_ROWS:
        raise SystemExit(f"REFUSED: pool has {len(rows)} rows")
    json.dump(rows, open(POOL, "w"))
    marks, b, rnd, groups, keyp, keyt = _arrays(rows)
    ref = b & np.isin(rnd, TECH)
    res = {"rows": len(rows), "ref_rows": int(ref.sum()), "keep": KEEP, "repeats": REPEATS}
    for g in ("passage", "pair", "lemma"):
        res[g] = {"q50": _cv(ArmQ, rows, g, REPEATS), "r50": _cv(P12.ArmP, rows, g, REPEATS)}
        res[g]["paired_gain"] = float(np.mean(np.array(res[g]["q50"]["good_rate_by_repeat"])
                                              - np.array(res[g]["r50"]["good_rate_by_repeat"])))
    pq = res["passage"]["q50"]["all"]
    res["go_conditions"] = {
        "passage_cv_good_ge_0.90": pq["good_rate"] >= 0.90,
        "passage_cv_keep_ge_0.45": pq["keep_share"] >= 0.45,
        "passage_cv_gain_ge_-0.01": res["passage"]["paired_gain"] >= -0.01,
        "pair_cv_gain_ge_0.01": res["pair"]["paired_gain"] >= 0.01,
        "lemma_cv_gain_ge_0.01": res["lemma"]["paired_gain"] >= 0.01}
    res["go"] = bool(all(res["go_conditions"].values()))
    pr = res["passage"]["r50"]["all"]
    res["r50_check"] = {"passage_cv_good_ge_0.90": pr["good_rate"] >= 0.90, "passage_cv_keep_ge_0.45": pr["keep_share"] >= 0.45}
    res["r50_go"] = bool(all(res["r50_check"].values()))
    res["pool_sha256"] = _sha(POOL)
    res["source_sha256"] = {p: _sha(p) for p in SOURCES}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "source_sha256"}, indent=1))
    # r50 (arm p refitted on all 1,259 rows) and q50 are rated only if their checks pass; the frozen
    # arms (b, s, gs, p, p50) are rated either way (the genre test needs no refit)
    thr = {}
    for name, cls, path in (("r50", P12.ArmP, MODEL_R), ("q50", ArmQ, MODEL_Q)):
        full = np.mean([P14.oof(cls, rows, groups, P12.SEED + k) for k in range(REPEATS)], axis=0)
        thr[name] = float(np.quantile(full[ref], 1 - KEEP))
        pickle.dump(cls(rows), open(path, "wb"))
    pre19 = json.load(open(P19.PRECOMMIT))
    arms = [a for a in ARMS if (a != "q50" or res["go"]) and (a != "r50" or res["r50_go"])]
    json.dump({"go": res["go"], "r50_go": res["r50_go"], "arms": arms,
               "thresholds": dict(thr, p50=pre19["thresholds"]["p50"], p=pre19["thresholds"]["p"],
                                  s=pre19["thresholds"]["s"]),
               "keep": KEEP, "model_q_sha256": _sha(MODEL_Q), "model_r_sha256": _sha(MODEL_R),
               "model_p50_sha256": _sha(P19.MODEL), "r12_model_sha256": _sha(P12.MODEL), "pool_sha256": _sha(POOL),
               "sense_table_sha256": _sha(R.TABLE), "arm_s_model_sha256": _sha(S.MODEL),
               "genre_manifest_sha256": _sha(GEN.GENRE / "manifest.json"),
               "genre_corpus_sha256": {g: _sha(GEN.GENRE / f"{g}.jsonl.gz") for g in GENRES},
               "source_sha256": {p: _sha(p) for p in SOURCES},
               "windows": {k: {"seed": v[1], "per_source": v[2]} for k, v in WINDOWS.items()},
               "joshua_page": {"best_arm_rule": "q50 if q50 was rated and meets all its round-21 bars, else p50",
                               "best_arm_swaps": J_BEST, "arm_b_rejected_swaps": J_B},
               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
    print(f"pre-committed thresholds {thr}; go={res['go']}")
    print("COMPLETE")


def load_models():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL_Q) != pre["model_q_sha256"] or _sha(MODEL_R) != pre["model_r_sha256"] \
            or _sha(P19.MODEL) != pre["model_p50_sha256"] or _sha(P12.MODEL) != pre["r12_model_sha256"] \
            or _sha(R.TABLE) != pre["sense_table_sha256"] or _sha(S.MODEL) != pre["arm_s_model_sha256"]:
        raise SystemExit("REFUSED: models or sense table differ from the round-21 pre-commitment")
    return {"q50": pickle.load(open(MODEL_Q, "rb")), "r50": pickle.load(open(MODEL_R, "rb")),
            "p50": pickle.load(open(P19.MODEL, "rb")), "p": pickle.load(open(P12.MODEL, "rb"))}, pre


# --- fresh windows ---------------------------------------------------------------------

def _docs(src):
    path = (GEN.GENRE if src in GENRES else R.CORPUS) / f"{src}.jsonl.gz"
    docs = {}
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        docs.setdefault(r["doc"], []).append(r["s"])
    return docs


def panel(window):
    """PASSAGE_SENTENCES consecutive clean sentences, one passage per document, from a seeded
    shuffle of each source. Excludes every document used in rounds 11-13 and 17-20 (and, for
    Joshua's window, round 21), and any passage sharing a word 8-gram with any earlier panel or
    window (rounds 4, 5, 8, 11-13, 17-20, and 21 for Joshua's), any rated development sentence or
    test-set v3's panel file (exclusion only). Rewrite-corpus overlap is checked at apply."""
    from cleartext.learned_words.rewrite import tokens

    d, seed, per = WINDOWS[window]
    path = d / "passages.json"
    if path.exists():
        raise SystemExit(f"REFUSED: {path} exists")
    prev = []
    for p in (R.PANEL, P12.PANEL, P13.PANEL, P17.PANEL, P18.PANEL, P19.PANEL, P20.PANEL):
        prev += json.load(open(p))
    if window == "joshua":
        prev += json.load(open(DIR / "passages.json"))
    used = [x["original"] for x in prev]
    for p in ("panel-r4", "panel-r5", "panel-r8"):
        used += [x["original"] for x in json.load(open(STATE / p / "passages.json"))]
    used += [s["before"] for s in R.dev_swaps()]
    used += [x["original"] for x in json.load(open(V3_PANEL))]
    grams = P12._grams(used)
    prev_docs = {(x["domain"], x["source_doc"]) for x in prev}
    out = []
    for src, want in per.items():
        docs = _docs(src)
        order = sorted(docs)
        rng = random.Random(f"{seed}:{src}")
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
            out.append({"id": f"{window}-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": doc,
                        "first_sentence": i, "original": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({"window": window, "passages": len(out), "by_source": Counter(p["domain"] for p in out)}), flush=True)


def _overlap(window, pan):
    from cleartext.learned_words.rewrite import tokens

    from .learned_words_rewrite import pairs

    path = WINDOWS[window][0] / "panel-overlap.json"
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
    """Round 19's apply on the round-21 window, then on Joshua's window (checkpointed)."""
    from cleartext.learned_words import senses as SN

    from .learned_splits import sent
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    R._threads(argv)
    load_models()
    deadline = time.monotonic() + R._seconds(argv, 1200)
    clf_s = sn = T = ids = model = simpler = None
    for window in WINDOWS:
        d = WINDOWS[window][0]
        if not (d / "passages.json").exists():
            panel(window)
        pan = json.load(open(d / "passages.json"))
        skip = _overlap(window, pan)
        out = d / "apply.jsonl"
        done = {r["pid"] for r in read_jsonl(out)}
        todo = [p for p in pan if p["id"] not in done and p["id"] not in skip]
        print(f"{window}: {len(done)} done, {len(todo)} to do, {len(skip)} skipped for rewrite-corpus overlap", flush=True)
        if todo and clf_s is None:
            clf_s, _ = S.load_filter()
            sn = SN.load(R.TABLE)
            _, T, ids, _ = load()
            model = pickle.load(open(STATE / "model.pkl", "rb"))
            simpler = pickle.load(open(SIMPLER, "rb"))
        with open(out, "a") as f:
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
        left = len({p["id"] for p in pan} - skip - {r["pid"] for r in read_jsonl(out)})
        if left:
            print(f"PARTIAL {window}: {left} left", flush=True)
            return
    print("COMPLETE", flush=True)


def _rows_q(swaps):
    return [{"key": list(P12.pair_key(s["word"], s["rep"], s["pos"])), "stack": s["stack"], "cover": s["cover"] or 0.0,
             "good": 0, "x31": s["x31"], "ctx": context(s["before"], s["after"], s["rep"])} for s in swaps]


def picks(rec, models, thr, arms, choose):
    """b, s, gs, p as before; p50: round 19's frozen model; r50: arm p refitted on 1,259 rows; q50:
    arm q. Filters apply before the per-passage limits."""
    a = [s for s in rec["all"] if s["margin"] >= S.MIN_MARGIN]
    g = [s for s in rec["sense"] if s["margin"] >= S.MIN_MARGIN]
    if a:
        rq = _rows_q(a)
        for name in ("p", "p50", "r50"):
            for s, x in zip(a, P12.score_p(models[name], a)):
                s["sc_" + name] = float(x)
        for s, x in zip(a, models["q50"].score(rq)):
            s["sc_q50"] = float(x)
    out = {"b": choose(a), "s": choose([s for s in a if s["stack"] >= thr["s"]]),
           "gs": choose([s for s in g if s["stack"] >= thr["s"]])}
    for name in ("p", "p50", "r50", "q50"):
        out[name] = choose([s for s in a if s["sc_" + name] >= thr[name]])
    return {k: v for k, v in out.items() if k in arms}


def realized(s):
    """The displayed sentence: the swap with the indefinite article repaired (surface realization)."""
    from cleartext.learned_words.realize import fix_article

    return fix_article(s["after"], start_of(s["before"], s["after"]), s["rep"])


def _items(window, models, pre):
    from .learned_words_apply import _skey, choose

    items = {}
    for r in read_jsonl(WINDOWS[window][0] / "apply.jsonl"):
        for arm, chosen in picks(r, models, pre["thresholds"], pre["arms"], choose).items():
            for s in chosen:
                k = _skey(s)
                it = items.setdefault(k, dict({x: v for x, v in s.items() if x != "x31"}, key=k, arms=[],
                                              domain=r["domain"], shown=realized(s)))
                if arm not in it["arms"]:
                    it["arms"].append(arm)
    return sorted(items.values(), key=lambda x: x["key"])


def page(argv):
    models, pre = load_models()
    items = _items("r21", models, pre)
    rng = random.Random(1221)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r21") for x in items] + [dict(x, source="old", old_mark=x["mark"], shown=x["after"])
                                                     for x in old[:P12.CONSISTENCY_ITEMS]]
    RATING.mkdir(parents=True, exist_ok=True)
    for r in range(1, P12.RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r21" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    for r in range(1, P12.RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1221:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["shown"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r21_items": len(items), "items": len(base), "realized": sum(x["shown"] != x["after"] for x in items),
                      "per_arm": Counter(a for x in items for a in x["arms"]),
                      "per_domain": Counter(x["domain"] for x in items)}, indent=1))


def score(argv):
    from .learned_words_study import wilson

    models, pre = load_models()
    rows = pool()
    pool_pairs = {tuple(r["key"]) for r in rows}
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
    recs = read_jsonl(DIR / "apply.jsonl")
    sents = {g: sum(r["sentences"] for r in recs if _part(r["domain"]) == g) for g in ("genre", "tech")}
    sents["all"] = sum(sents.values())
    arms = pre["arms"]

    def pk(m):
        s = key[m["key"]]
        return tuple(P12.pair_key(s["word"], s["rep"], s["pos"]))

    def block(marks, got, nsent):
        ms = [marks[m["id"]] for m in got]
        c = Counter(ms)
        un = [marks[m["id"]] for m in got if pk(m) not in pool_pairs]
        return {"swaps": len(ms), **c, "good_rate": c["good"] / max(1, len(ms)), "good_ci95": wilson(c["good"], len(ms)),
                "meaning_rate": c["meaning"] / max(1, len(ms)), "per_100_sentences": 100 * len(ms) / max(1, nsent),
                "distinct_pairs": len({pk(m) for m in got}),
                "unseen_pairs": {"swaps": len(un), "good": un.count("good"), "meaning": un.count("meaning")}}

    def arm_stats(marks):
        res = {}
        for arm in arms:
            got = [m for m in mp if m["source"] == "r21" and arm in m["arms"]]
            res["arm_" + arm] = dict(block(marks, got, sents["all"]), **{
                part: block(marks, [m for m in got if _part(key[m["key"]]["domain"]) == part], sents[part])
                for part in ("genre", "tech")}, by_domain={
                d: {"swaps": sum(key[m["key"]]["domain"] == d for m in got),
                    "good": sum(key[m["key"]]["domain"] == d and marks[m["id"]] == "good" for m in got)}
                for d in ("news", "wiki", "fedreg", "openfda")})
        return res

    out = {"items": len(ids), "raters": P12.RATERS, "passages": len(recs), "sentences": sents,
           "majority": arm_stats(maj), "per_rater": [arm_stats(m) for m in raters]}
    M = out["majority"]
    for part in ("all", "genre", "tech"):
        get = (lambda a: M["arm_" + a]["swaps"]) if part == "all" else (lambda a: M["arm_" + a][part]["swaps"])
        out[f"yield_vs_b_{part}"] = {a: get(a) / max(1, get("b")) for a in arms}

    def bars(arm, part):
        v = M["arm_" + arm] if part == "all" else M["arm_" + arm][part]
        y = out[f"yield_vs_b_{part}"][arm]
        return {"good_ge_0.90": v["good_rate"] >= 0.90, "meaning_le_0.08": v["meaning_rate"] <= 0.08,
                "yield_ge_0.40": y >= 0.40}

    out["success_p50_genre"] = bars("p50", "genre")
    if "q50" in arms:
        q, p = M["arm_q50"], M["arm_p50"]
        qb = bars("q50", "all")
        qb["unseen_swaps_ge_1.25x_p50"] = q["unseen_pairs"]["swaps"] >= 1.25 * max(1, p["unseen_pairs"]["swaps"])
        qb["unseen_good_ge_0.85"] = q["unseen_pairs"]["good"] >= 0.85 * max(1, q["unseen_pairs"]["swaps"])
        out["success_q50"] = qb
    out["realized_items"] = sum(key[m["key"]]["shown"] != key[m["key"]]["after"] for m in mp if m["source"] == "r21")
    r21 = [x["id"] for x in mp if x["source"] == "r21"]
    out["unanimous_share"] = sum(len({rt[i] for rt in raters}) == 1 for i in r21) / max(1, len(r21))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r21) / max(1, len(r21))
                                 for a in range(P12.RATERS) for c in range(a + 1, P12.RATERS)}
    old = [x for x in mp if x["source"] == "old"]
    out["consistency_with_old"] = [sum(x["old_mark"] == rt[x["id"]] for x in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _part(domain):
    return "genre" if domain in GENRES else "tech"


# --- Joshua's blind marking page -------------------------------------------------------

def joshua_page(argv):
    """About J_BEST swaps from the best arm and J_B arm-b swaps that arm rejects, from Joshua's
    window (never shown to an AI rater), mixed in one seeded shuffle with no scores or arm labels.
    The key stays in the state directory; marks download as CSV. His marks decide adoption and are
    never used to train, tune or choose."""
    models, pre = load_models()
    res = json.load(open(RESULT))
    best = "q50" if "q50" in pre["arms"] and all(res.get("success_q50", {"x": False}).values()) else "p50"
    items = _items("joshua", models, pre)
    rng = random.Random(1231)
    a = [x for x in items if best in x["arms"]]
    rest = [x for x in items if "b" in x["arms"] and best not in x["arms"]]
    rng.shuffle(a)
    rng.shuffle(rest)
    chosen = [dict(x, group="best") for x in a[:J_BEST]] + [dict(x, group="b_rejected") for x in rest[:J_B]]
    rng.shuffle(chosen)
    for n, x in enumerate(chosen, 1):
        x["id"] = f"j{n}"
    JDIR.mkdir(parents=True, exist_ok=True)
    if (JDIR / "key.json").exists() and json.load(open(JDIR / "key.json"))["items"] != [x["key"] for x in chosen]:
        raise SystemExit("REFUSED: a different key exists")
    window_counts = {"best_arm_swaps": len(a), "arm_b_swaps": sum("b" in x["arms"] for x in items),
                     "arm_b_rejected_by_best": len(rest),
                     "by_domain": Counter(x["domain"] for x in items if "b" in x["arms"])}
    json.dump({"best_arm": best, "items": [x["key"] for x in chosen], "window": window_counts,
               "swaps": [{k: v for k, v in x.items() if k not in ("x31",)} for x in chosen]},
              open(JDIR / "key.json", "w"), indent=1)
    data = [{"id": x["id"], "before": x["before"], "after": x["shown"], "word": x["word"], "rep": x["rep"]}
            for x in chosen]
    JPAGE.parent.mkdir(parents=True, exist_ok=True)
    JPAGE.write_text(_html(data))
    print(json.dumps({"best_arm": best, "page": str(JPAGE), "items": len(chosen), **window_counts}, indent=1))


def _mark_word(text, start, word):
    return html.escape(text[:start]) + "<mark>" + html.escape(word) + "</mark>" + html.escape(text[start + len(word):])


def _html(data):
    """Round 4's page style: one card per swap, three buttons, marks kept in the browser, CSV download."""
    cards = []
    for n, x in enumerate(data, 1):
        b0 = start_of(x["before"], x["after"])
        a0 = x["after"].find(x["rep"], max(0, b0 - 3))
        before = _mark_word(x["before"], b0, x["word"]) if x["before"][b0:b0 + len(x["word"])] == x["word"] \
            else html.escape(x["before"])
        after = _mark_word(x["after"], a0, x["rep"]) if a0 >= 0 else html.escape(x["after"])
        cards.append(f'<div class="card" data-id="{x["id"]}"><div class="n">{n} <b>{html.escape(x["word"])} &rarr; '
                     f'{html.escape(x["rep"])}</b></div><div class="lbl">Before</div><p>{before}</p>'
                     f'<div class="lbl">After</div><p>{after}</p><div class="btns"><button data-v="good">Good</button>'
                     f'<button data-v="meaning">Changes meaning</button><button data-v="notsimpler">Not simpler</button>'
                     f'</div></div>')
    return ("""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rate Word Swaps</title><style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--acc:#2563eb;--good:#16a34a;--bad:#dc2626;--meh:#d97706}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#999;--line:#2a2a2a}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0} .intro{color:var(--mut)}
.card{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:12px 0}
.n{font-size:13px;color:var(--mut);margin-bottom:6px} .n b{color:var(--fg)}
.lbl{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em} p{margin:2px 0 10px}
mark{background:#fde68a;color:#111;border-radius:3px;padding:0 2px}
.btns{display:flex;gap:8px;flex-wrap:wrap} button{font:inherit;font-size:14px;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:none;color:var(--fg);cursor:pointer}
.card[data-mark] {opacity:.55} .card[data-mark]:hover{opacity:1}
.card[data-mark=good] button[data-v=good]{background:var(--good);color:#fff;border-color:var(--good)}
.card[data-mark=meaning] button[data-v=meaning]{background:var(--bad);color:#fff;border-color:var(--bad)}
.card[data-mark=notsimpler] button[data-v=notsimpler]{background:var(--meh);color:#fff;border-color:var(--meh)}
.bar{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;display:flex;justify-content:space-between;align-items:center;gap:8px}
.bar button{background:var(--acc);color:#fff;border-color:var(--acc)}
</style></head><body><main>
<h1>Rate the word swaps (round 21)</h1>
<p class="intro">Each card shows a sentence before and after one word was replaced, from news, encyclopedia and
government or drug-label text. Tap <b>Good</b> only if the highlighted replacement is clearly simpler for an ordinary
adult reader and the sentence keeps its meaning and still reads naturally. <b>Not simpler</b>: the meaning is fine but
the new word is no easier. <b>Changes meaning</b>: the meaning changes, or the sentence now reads wrong. Your marks save
in this browser as you go. When you're done, tap <b>Download marks</b> at the top (it saves
<code>cleartext-marks-learned-swaps-r21.csv</code>). Scores are hidden.</p>
<div class="bar"><span id="count"></span><button id="dl">Download marks</button></div>
""" + "".join(cards) + """</main><script>
const KEY='cleartext-rate-learned-r21-20261122';const cs=[...document.querySelectorAll('.card')];
let m={};try{m=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function paint(){cs.forEach(c=>{const v=m[c.dataset.id];if(v)c.dataset.mark=v;else delete c.dataset.mark});
document.getElementById('count').textContent=Object.keys(m).length+' of '+cs.length+' rated';
try{localStorage.setItem(KEY,JSON.stringify(m))}catch(e){}}
cs.forEach(c=>c.querySelectorAll('button').forEach(b=>b.onclick=()=>{const id=c.dataset.id;
if(m[id]===b.dataset.v)delete m[id];else m[id]=b.dataset.v;paint()}));
document.getElementById('dl').onclick=()=>{const rows=['id,mark',...cs.map(c=>c.dataset.id+','+(m[c.dataset.id]||''))].join('\\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows+'\\n'],{type:'text/csv'}));a.download='cleartext-marks-learned-swaps-r21.csv';a.click()};
paint();
</script></body></html>
""")


def main(argv):
    cmd = {"unseen": unseen, "train": train, "apply": apply, "page": page, "score": score, "joshua-page": joshua_page}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r21 unseen|train|apply|page|score|joshua-page")
