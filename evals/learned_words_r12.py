"""Round 12: learned pair priors and sense cover on top of the arm-s score (see ROUND12.md).

  learned-words r12 train                    CV gate; final fit and pre-commit if go
  learned-words r12 apply [--max-seconds N]  fresh panel (once), stage, arms s/gs features (checkpointed)
  learned-words r12 page | score             three blind rater files; majority-vote scoring

Arm p scores each arm-b swap with a 4-feature logistic: logit of the arm-s score, the target's
in-context sense cover, and smoothed Good rates of the (target, replacement) lemma pair and of the
target lemma, learned from earlier blind ratings. Arm t is the arm-s score alone at the same keep.
Raw text, pools and marks stay in the state directory (r12/).
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

from . import learned_words_r9 as R
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r12"
POOL = DIR / "pool.json"
MODEL = DIR / "model.pkl"
PANEL = DIR / "passages.json"
OVERLAP = DIR / "panel-overlap.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r12-precommit.json"
CV_REPORT = REPORT / "r12-cv.json"
RESULT = REPORT / "round12.json"
SEED = 20261212
SMOOTH = 2.0  # pseudo-counts at the pool's Good rate
KEEP = 0.50  # arms p and t: share of fresh-like arm-b rows kept
KEEP_WIDE = 0.65  # arm p65: the same model at a wider keep (secondary, maps the trade-off)
REPEATS = 20
FOLDS = 5
PER_SOURCE = 45
PASSAGE_SENTENCES = 4
CONSISTENCY_ITEMS = 20
RATERS = 3
EPS = 1e-3
ARMS = ("b", "s", "gs", "t", "p", "p65")
SOURCES = ("evals/learned_words_r12.py", "evals/learned_words_r9.py", "evals/learned_words_stacker.py",
           "evals/learned_words_apply.py", "models/src/cleartext/learned_words/senses.py")


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _logit(p):
    p = min(max(float(p), EPS), 1 - EPS)
    return float(np.log(p / (1 - p)))


def pair_key(word, rep, pos):
    """(target lemma, replacement lemma), with the stage's own lemmatizer."""
    from cleartext.learned_words import features as F

    u = F.UPOS.get(pos)
    return F.lemma_of(word, u), F.lemma_of(rep, u)


class Priors:
    """Smoothed Good rates per (target, replacement) lemma pair and per target lemma."""

    def __init__(self, keys, good, a=SMOOTH):
        self.a, self.m = a, float(np.mean(good))
        self.pair, self.tgt = {}, {}
        for k, g in zip(keys, good):
            k = tuple(k)
            p, t = self.pair.setdefault(k, [0, 0]), self.tgt.setdefault(k[0], [0, 0])
            p[0] += int(g)
            p[1] += 1
            t[0] += int(g)
            t[1] += 1

    def rates(self, key, own=None):
        """(pair rate, target rate, pair count); own = the row's own label, removed (leave one out)."""
        key = tuple(key)
        g, n = self.pair.get(key, (0, 0))
        gt, nt = self.tgt.get(key[0], (0, 0))
        if own is not None:
            g, n, gt, nt = g - int(own), n - 1, gt - int(own), nt - 1
        return (g + self.a * self.m) / (n + self.a), (gt + self.a * self.m) / (nt + self.a), n

    def seen(self, key):
        return tuple(key) in self.pair


def design(rows, pri, loo):
    out = []
    for r in rows:
        pp, pt, _ = pri.rates(r["key"], r["good"] if loo else None)
        out.append([_logit(r["stack"]), float(r["cover"]), _logit(pp), _logit(pt)])
    return np.array(out, float)


class ArmP:
    """Arm p: priors from the training rows plus a standardized logistic (C = 0.1) on four features."""

    def __init__(self, rows):
        self.pri = Priors([r["key"] for r in rows], [r["good"] for r in rows])
        self.clf = S._fit(design(rows, self.pri, loo=True), np.array([r["good"] for r in rows]))

    def score(self, rows):
        return self.clf.predict_proba(design(rows, self.pri, loo=False))[:, 1]


# --- training pool -------------------------------------------------------------------

def _dev_rows():
    """The 371 round-9 development rows. Their arm-s score is out of fold for the 334 rows the
    committed arm-s model was trained on (mean over REPEATS lemma-grouped 5-fold runs) and the
    committed model's score for the round-8 rows. Cover is the round-9 `sense_t_cov`."""
    Xs, _, marks, b, groups, swaps = R.dataset()
    y = (marks == "good").astype(int)
    r8 = np.array([s["round"] == "r8" for s in swaps])
    clf, _ = S.load_filter()
    st = np.zeros(len(y))
    st[~r8] = np.mean([S._oof(Xs[~r8], y[~r8], groups[~r8], SEED + k) for k in range(REPEATS)], axis=0)
    st[r8] = clf.predict_proba(Xs[r8])[:, 1]
    rows = {r["src"]: r["row"] for r in read_jsonl(R.FEATURES)}
    return [{"src": s["src"], "round": s["round"], "group": f"{s['round']}:{s['pid']}", "word": s["word"],
             "rep": s["rep"], "pos": s["pos"], "mark": s["mark"], "b": "b" in s["arms"], "stack": float(st[i]),
             "cover": float(rows[s["src"]]["new"]["sense_t_cov"])} for i, s in enumerate(swaps)]


def _r11_rows():
    """The 135 round-11 rated swaps with their majority marks (the committed arm-s score, the
    in-context sense cover from apply11)."""
    mp = json.load(open(R.RATING11 / "blind-map.json"))
    raters = [{x["id"]: x["mark"] for x in csv.DictReader(open(R.RATING11 / f"marks-{r}.csv"))}
              for r in range(1, RATERS + 1)]
    key = {s["key"]: s for s in json.load(open(R.RATING11 / "key.json"))["swaps"]}
    out = []
    for m in mp:
        if m["source"] != "r11":
            continue
        s = key[m["key"]]
        out.append({"src": f"r11:{m['key']}", "round": "r11", "group": f"r11:{s['pid']}", "word": s["word"],
                    "rep": s["rep"], "pos": s["pos"], "mark": R.majority([rt[m["id"]] for rt in raters]),
                    "b": "b" in m["arms"], "stack": float(s["stack"]), "cover": float(s["cover"] or 0.0)})
    return out


def pool():
    rows = _dev_rows() + _r11_rows()
    for r in rows:
        r["key"] = list(pair_key(r["word"], r["rep"], r["pos"]))
        r["good"] = int(r["mark"] == "good")
    return rows


# --- cross-validation ----------------------------------------------------------------

def _folds(groups, seed):
    from sklearn.model_selection import GroupKFold

    z = np.zeros(len(groups))
    return GroupKFold(FOLDS, shuffle=True, random_state=seed).split(z, z, groups)


def oof_p(rows, groups, seed):
    """Out-of-fold arm-p scores; held-out rows see priors from the training folds only."""
    p = np.zeros(len(rows))
    for tr, te in _folds(groups, seed):
        m = ArmP([rows[i] for i in tr])
        p[te] = m.score([rows[i] for i in te])
    return p


def nested(rows, groups, ref, seed, keep=KEEP):
    """Outer out-of-fold (p score, p keep, t keep). Each threshold keeps `keep` of the training
    folds' reference rows (fresh-like arm-b rows): inner out-of-fold scores for p, the arm-s
    score itself for t."""
    n = len(rows)
    score, kp, kt = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    stack = np.array([r["stack"] for r in rows])
    for tr, te in _folds(groups, seed):
        sub = [rows[i] for i in tr]
        inner = oof_p(sub, groups[tr], seed + 1)
        tp = np.quantile(inner[ref[tr]], 1 - keep)
        tt = np.quantile(stack[tr][ref[tr]], 1 - keep)
        score[te] = ArmP(sub).score([rows[i] for i in te])
        kp[te] = score[te] >= tp
        kt[te] = stack[te] >= tt
    return score, kp, kt


def _stats(keep, marks, on):
    k = keep & on
    c = Counter(marks[k])
    return {"kept": int(k.sum()), "keep_share": float(k.sum() / max(1, on.sum())),
            "good_rate": float(c["good"] / max(1, k.sum())), "meaning": int(c["meaning"]),
            "notsimpler": int(c["notsimpler"])}


def _auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def train(argv):
    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    DIR.mkdir(exist_ok=True)
    rows = pool()
    marks = np.array([r["mark"] for r in rows])
    y = np.array([r["good"] for r in rows])
    b = np.array([r["b"] for r in rows])
    r11 = np.array([r["round"] == "r11" for r in rows])
    ref = b & r11
    if len(rows) != 506 or ref.sum() != 126 or b.sum() != 315:
        raise SystemExit(f"REFUSED: pool has {len(rows)} rows, {b.sum()} arm-b, {ref.sum()} round-11 arm-b")
    json.dump(rows, open(POOL, "w"))
    stack = np.array([r["stack"] for r in rows])
    _, ts = S.load_filter()
    cover = np.array([r["cover"] for r in rows])
    res = {"rows": len(rows), "arm_b_rows": int(b.sum()), "r11_arm_b_rows": int(ref.sum()), "keep": KEEP,
           "repeats": REPEATS, "smooth": SMOOTH,
           "base_r11_b": _stats(np.ones(len(rows), bool), marks, ref),
           "arm_s_r11_b": _stats(stack >= ts, marks, ref),
           "gs_rule_r11_b": _stats((stack >= ts) & (cover >= R.SENSE_COVER), marks, ref)}

    # A. passage-grouped nested CV (fresh-like: new passages, earlier ratings available as priors)
    groups = np.array([r["group"] for r in rows])
    runs = []
    for k in range(REPEATS):
        score, kp, kt = nested(rows, groups, ref, SEED + 10 * k)
        runs.append({"p": _stats(kp, marks, ref), "t": _stats(kt, marks, ref), "p_all_b": _stats(kp, marks, b),
                     "t_all_b": _stats(kt, marks, b), "auc_p_r11_b": _auc(y[ref], score[ref]),
                     "auc_t_r11_b": _auc(y[ref], stack[ref])})

    def mean(key, f):
        return float(np.mean([r[key][f] for r in runs]))

    res["passage_cv"] = {a: {f: mean(a, f) for f in ("kept", "keep_share", "good_rate", "meaning", "notsimpler")}
                         for a in ("p", "t", "p_all_b", "t_all_b")}
    res["passage_cv"]["paired_gain_r11_b"] = float(np.mean([r["p"]["good_rate"] - r["t"]["good_rate"] for r in runs]))
    res["passage_cv"]["good_rate_min_max_p"] = [float(min(r["p"]["good_rate"] for r in runs)),
                                                float(max(r["p"]["good_rate"] for r in runs))]
    res["passage_cv"]["auc_p_r11_b"] = float(np.mean([r["auc_p_r11_b"] for r in runs]))
    res["passage_cv"]["auc_t_r11_b"] = runs[0]["auc_t_r11_b"]

    # B. time split: fit on rounds 3-8, score round 11 (thresholds from rounds 3-8 arm-b rows)
    dev = ~r11
    d_rows = [r for r, x in zip(rows, dev) if x]
    inner = np.mean([oof_p(d_rows, groups[dev], SEED + k) for k in range(REPEATS)], axis=0)
    tp = np.quantile(inner[b[dev]], 1 - KEEP)
    tt = np.quantile(stack[dev][b[dev]], 1 - KEEP)
    sc = np.zeros(len(rows))
    sc[r11] = ArmP(d_rows).score([r for r, x in zip(rows, r11) if x])
    res["time_split"] = {"p": _stats(r11 & (sc >= tp), marks, ref), "t": _stats(r11 & (stack >= tt), marks, ref),
                         "auc_p_r11_b": _auc(y[ref], sc[ref]), "threshold_p": float(tp), "threshold_t": float(tt)}

    # C. diagnostic: lemma-grouped CV, where no held-out pair or target has a prior
    lem = np.array([r["key"][0] for r in rows])
    lruns = []
    for k in range(REPEATS):
        score, kp, kt = nested(rows, lem, ref, SEED + 10 * k)
        lruns.append({"p": _stats(kp, marks, ref), "t": _stats(kt, marks, ref)})
    res["lemma_cv_diagnostic"] = {a: {f: float(np.mean([r[a][f] for r in lruns]))
                                      for f in ("keep_share", "good_rate", "meaning", "notsimpler")} for a in ("p", "t")}

    # How often a round-11 arm-b row's pair had been rated in rounds 3-8
    dev_keys = {tuple(r["key"]) for r in d_rows}
    res["r11_b_pairs_seen_in_dev"] = int(sum(tuple(r["key"]) in dev_keys for r, x in zip(rows, ref) if x))
    pc = res["passage_cv"]
    res["go_conditions"] = {"cv_good_ge_0.92": pc["p"]["good_rate"] >= 0.92,
                            "cv_paired_gain_ge_0.03": pc["paired_gain_r11_b"] >= 0.03,
                            "cv_keep_ge_0.45": pc["p"]["keep_share"] >= 0.45,
                            "time_split_good_ge_0.90_keep_ge_0.40": res["time_split"]["p"]["good_rate"] >= 0.90
                            and res["time_split"]["p"]["keep_share"] >= 0.40}
    res["go"] = bool(all(res["go_conditions"].values()))
    res["pool_sha256"] = _sha(POOL)
    res["source_sha256"] = {p: _sha(p) for p in SOURCES}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "source_sha256"}, indent=1))
    if res["go"]:
        full = np.mean([oof_p(rows, groups, SEED + k) for k in range(REPEATS)], axis=0)
        thr = {"p": float(np.quantile(full[ref], 1 - KEEP)), "p65": float(np.quantile(full[ref], 1 - KEEP_WIDE)),
               "t": float(np.quantile(stack[ref], 1 - KEEP)), "s": float(ts)}
        pickle.dump(ArmP(rows), open(MODEL, "wb"))
        json.dump({"thresholds": thr, "keep": KEEP, "keep_wide": KEEP_WIDE, "smooth": SMOOTH,
                   "sense_cover_gs": R.SENSE_COVER, "model_sha256": _sha(MODEL), "pool_sha256": _sha(POOL),
                   "sense_table_sha256": _sha(R.TABLE), "arm_s_model_sha256": _sha(S.MODEL),
                   "arm_s_precommit_sha256": _sha(S.PRECOMMIT), "source_sha256": {p: _sha(p) for p in SOURCES},
                   "panel_seed": SEED, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
                  open(PRECOMMIT, "w"), indent=1)
        print(f"pre-committed thresholds {thr}")
    print("COMPLETE")


def load_p():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL) != pre["model_sha256"] or _sha(R.TABLE) != pre["sense_table_sha256"] \
            or _sha(S.MODEL) != pre["arm_s_model_sha256"]:
        raise SystemExit("REFUSED: arm p model, sense table or arm s differ from the round-12 pre-commitment")
    return pickle.load(open(MODEL, "rb")), pre["thresholds"]


# --- fresh window ----------------------------------------------------------------------

def _grams(texts):
    from cleartext.learned_words.rewrite import tokens

    g = set()
    for t in texts:
        tt = tokens(t)
        g |= {tuple(tt[i:i + 8]) for i in range(len(tt) - 7)}
    return g


def panel():
    """PER_SOURCE passages of PASSAGE_SENTENCES clean consecutive sentences, one per document,
    from a new seeded shuffle; no document used in round 11 and no word 8-gram shared with the
    round-4/5/8/11 panels or any rated development sentence."""
    from cleartext.learned_words.rewrite import tokens

    if PANEL.exists():
        raise SystemExit(f"REFUSED: {PANEL} exists")
    r11 = json.load(open(R.PANEL))
    used = [x["original"] for x in r11]
    for p in ("panel-r4", "panel-r5", "panel-r8"):
        used += [x["original"] for x in json.load(open(STATE / p / "passages.json"))]
    used += [s["before"] for s in R.dev_swaps()]
    grams = _grams(used)
    r11_docs = {(x["domain"], x["source_doc"]) for x in r11}
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
            if (src, d) in r11_docs:
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
            out.append({"id": f"r12-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": d,
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


def apply(argv):
    """Round 11's two generators from one featurization per target (all candidates; sense cover
    >= 0.25), the arm-s score and the 31 arm-s features of every resulting swap."""
    from cleartext.learned_words import senses as SN

    from .learned_splits import sent
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    R._threads(argv)
    load_p()
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
                    A = np.array([[r["f"][k] for k in S.STAGE_KEYS] for r in rows], float)
                    spans = [dict(s, start=r["start"]) for s, r in zip(got, rows)]
                    X = np.nan_to_num(np.hstack([A, S.rewrite_features(T, ids, spans)]))
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


def score_p(model, swaps):
    rows = [{"key": pair_key(s["word"], s["rep"], s["pos"]), "stack": s["stack"], "cover": s["cover"] or 0.0,
             "good": 0} for s in swaps]
    return model.score(rows) if rows else np.zeros(0)


def picks(rec, model, thr, choose):
    """Every arm's swaps in one passage; filters apply before the per-passage limits.
    b: all candidates, margin. s: b + arm s at its committed threshold. gs: sense-limited
    candidates, margin, arm s. t: b + arm-s score at the stricter round-12 threshold.
    p, p65: b + arm-p score at its two pre-committed thresholds."""
    a = [s for s in rec["all"] if s["margin"] >= S.MIN_MARGIN]
    g = [s for s in rec["sense"] if s["margin"] >= S.MIN_MARGIN]
    pa = score_p(model, a)
    for s, x in zip(a, pa):
        s["p"] = float(x)
    return {"b": choose(a), "s": choose([s for s in a if s["stack"] >= thr["s"]]),
            "gs": choose([s for s in g if s["stack"] >= thr["s"]]),
            "t": choose([s for s in a if s["stack"] >= thr["t"]]),
            "p": choose([s for s in a if s["p"] >= thr["p"]]),
            "p65": choose([s for s in a if s["p"] >= thr["p65"]])}


def page(argv):
    from .learned_words_apply import _skey, choose

    model, thr = load_p()
    items = {}
    for r in read_jsonl(APPLY):
        for arm, chosen in picks(r, model, thr, choose).items():
            for s in chosen:
                k = _skey(s)
                it = items.setdefault(k, dict({x: v for x, v in s.items() if x != "x31"}, key=k, arms=[],
                                              domain=r["domain"]))
                it["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1212)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r12") for x in items] + [dict(x, source="old", old_mark=x["mark"])
                                                     for x in old[:CONSISTENCY_ITEMS]]
    RATING.mkdir(exist_ok=True)
    for r in range(1, RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r12" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    for r in range(1, RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1212:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r12_items": len(items), "items": len(base),
                      "per_arm": Counter(a for x in items for a in x["arms"])}, indent=1))


def score(argv):
    from .learned_words_study import wilson

    model, _ = load_p()
    mp = json.load(open(RATING / "blind-map.json"))
    ids = {m["id"] for m in mp}
    raters = []
    for r in range(1, RATERS + 1):
        marks = {x["id"]: x["mark"] for x in csv.DictReader(open(RATING / f"marks-{r}.csv"))}
        if set(marks) != ids or set(marks.values()) - set(R.SEVERITY):
            raise SystemExit(f"REFUSED: rater {r} marks are incomplete or use an unknown label")
        raters.append(marks)
    maj = {i: R.majority([m[i] for m in raters]) for i in ids}
    key = {s["key"]: s for s in json.load(open(RATING / "key.json"))["swaps"]}
    recs = read_jsonl(APPLY)
    sentences = sum(r["sentences"] for r in recs)

    def arm_stats(marks):
        res = {}
        for arm in ARMS:
            got = [m for m in mp if m["source"] == "r12" and arm in m["arms"]]
            ms = [marks[m["id"]] for m in got]
            c = Counter(ms)
            res["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(c["good"], len(ms)), "meaning_rate": c["meaning"] / max(1, len(ms)),
                                 "per_100_sentences": 100 * len(ms) / max(1, sentences),
                                 "distinct_pairs": len({tuple(pair_key(key[m["key"]]["word"], key[m["key"]]["rep"],
                                                                       key[m["key"]]["pos"])) for m in got})}
            unseen = [marks[m["id"]] for m in got
                      if not model.pri.seen(pair_key(key[m["key"]]["word"], key[m["key"]]["rep"], key[m["key"]]["pos"]))]
            res["arm_" + arm]["unseen_pairs"] = {"swaps": len(unseen), "good": unseen.count("good")}
        return res

    out = {"items": len(ids), "raters": RATERS, "passages": len(recs), "sentences": sentences,
           "majority": arm_stats(maj), "per_rater": [arm_stats(m) for m in raters]}
    b = out["majority"]["arm_b"]
    out["yield_vs_b"] = {a: out["majority"]["arm_" + a]["swaps"] / max(1, b["swaps"]) for a in ARMS}
    p = out["majority"]["arm_p"]
    out["success"] = bool(p["good_rate"] >= 0.90 and p["meaning_rate"] <= 0.08 and out["yield_vs_b"]["p"] >= 0.40)
    r12 = [m["id"] for m in mp if m["source"] == "r12"]
    out["unanimous_share"] = sum(len({m[i] for m in raters}) == 1 for i in r12) / max(1, len(r12))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r12) / max(1, len(r12))
                                 for a in range(RATERS) for c in range(a + 1, RATERS)}
    old = [m for m in mp if m["source"] == "old"]
    out["consistency_with_old"] = [sum(m["old_mark"] == rt[m["id"]] for m in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"train": train, "apply": apply, "page": page, "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r12 train|apply|page|score")
