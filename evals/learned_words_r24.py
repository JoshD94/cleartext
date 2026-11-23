"""Round 24: dependency-anchored frame features and public pair priors (see ROUND24.md).

  learned-words r24 fetch                        extend the genre corpora with titles after round 21's first 220
  learned-words r24 features [--max-seconds N]   frame features of the 1,454 rated rows (checkpointed)
  learned-words r24 explore                      development diagnostics (r24-explore.json); fits nothing kept
  learned-words r24 train                        CV gate; final fits and pre-commit if go
  learned-words r24 apply [--max-seconds N]      fresh panel (once), round 11's generators, new features (checkpointed)
  learned-words r24 page | score                 three blind rater files; majority-vote scoring

Raw text, pools, features and marks stay in the state directory (r24/).
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

from . import learned_words_genre as GEN
from . import learned_words_r12 as P12
from . import learned_words_r14 as P14
from . import learned_words_r19 as P19
from . import learned_words_r21 as Q
from . import learned_words_r22 as G
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r24"
POOL = DIR / "pool.json"
FEAT = DIR / "pool-frame.jsonl"
MODEL_F = DIR / "model-f.pkl"
PANEL = DIR / "passages.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r24-precommit.json"
CV_REPORT = REPORT / "r24-cv.json"
RESULT = REPORT / "round24.json"
SEED = 20261224
PER_SOURCE = {"news": ("news2", 45), "wiki": ("wiki2", 60), "openfda": ("openfda", 45)}
GENRES = ("news", "wiki")
ARMS = ("b", "p50", "qg", "f", "fg")
SOURCES = ("evals/learned_words_r24.py", "models/src/cleartext/learned_words/depframe.py",
           "models/src/cleartext/learned_words/pairlex.py") + G.SOURCES
POOL_ROWS = 1377 + 77
EXTRA = 300  # genre titles per site after the first GEN.CAP of round 21's seeded shuffle
EXT_NAMES = {"news": "news2", "wiki": "wiki2"}


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def pool():
    """Round 22's 1,377-row pool plus round 22's 77 rated swaps (with sentence text and context)."""
    rows = json.load(open(G.POOL))
    new = P14._rated(G.RATING, G.APPLY, "r22")
    key = {s["key"]: s for s in json.load(open(G.RATING / "key.json"))["swaps"]}
    for r in new:
        s = key[r["src"].split(":", 1)[1]]
        r["key"] = list(P12.pair_key(r["word"], r["rep"], r["pos"]))
        r["good"] = int(r["mark"] == "good")
        r["before"], r["after"], r["domain"] = s["before"], s["after"], s["domain"]
        r["ctx"] = Q.context(r["before"], r["after"], r["rep"])
        r["tech"], r["genre"] = False, True
    return rows + new


def fetch(argv):
    """Round 21's genre fetch, unchanged, for titles GEN.CAP .. GEN.CAP + EXTRA of the same seeded
    shuffle (so no document can repeat one of the first 220). Written to genre/news2 and wiki2;
    the round-21 files are untouched."""
    from . import build_testset_v3 as B

    grams = set()
    for p in json.load(open(GEN.V3)):
        t = GEN.tokens(p["original"])
        grams.update(tuple(t[i:i + 8]) for i in range(len(t) - 7))
    stats = {}
    deadline = time.monotonic() + 1500
    for name, (api, cat) in GEN.SITES.items():
        out = GEN.GENRE / f"{EXT_NAMES[name]}.jsonl.gz"
        done = GEN.GENRE / f"{EXT_NAMES[name]}.done"
        if done.exists():
            stats[name] = json.loads(done.read_text())
            continue
        titles = GEN._titles(api, cat, deadline)
        order = list(titles)
        random.Random(f"{GEN.SEED}:{name}").shuffle(order)
        picked, docs = order[GEN.CAP:GEN.CAP + EXTRA], []
        for i in range(0, len(picked), 20):
            got = GEN._texts(api, picked[i:i + 20])
            docs += [(t, got[t][0], got[t][1]) for t in picked[i:i + 20] if t in got]
        kept = dropped = 0
        with gzip.open(out, "wt") as w:
            for t, rev, paras in docs:
                for para in paras:
                    for x in B.sentences(B.clean(para)):
                        tt = GEN.tokens(x)
                        if len(tt) < 4:
                            continue
                        if any(tuple(tt[i:i + 8]) in grams for i in range(len(tt) - 7)):
                            dropped += 1
                            continue
                        kept += 1
                        w.write(json.dumps({"doc": t, "pageid": rev, "s": x}) + "\n")
        stats[name] = {"titles_listed": len(titles), "range": [GEN.CAP, GEN.CAP + EXTRA], "documents": len(docs),
                       "sentences": kept, "dropped_v3_8gram_overlap": dropped,
                       "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        done.write_text(json.dumps(stats[name]))
        print(name, stats[name], flush=True)
    json.dump(stats, open(GEN.GENRE / "manifest-r24.json", "w"), indent=1)
    print("COMPLETE")


def features(argv):
    """Dependency-anchored frame features for every pool row (checkpointed by src)."""
    from cleartext.features import nlp
    from cleartext.learned_words import depframe as DF

    from . import learned_words_r9 as R

    R._threads(argv)
    deadline = time.monotonic() + R._seconds(argv, 1500)
    DIR.mkdir(parents=True, exist_ok=True)
    rows = pool()
    if len(rows) != POOL_ROWS:
        raise SystemExit(f"REFUSED: pool has {len(rows)} rows")
    if not POOL.exists():
        json.dump(rows, open(POOL, "w"))
    done = {r["src"] for r in read_jsonl(FEAT)} if FEAT.exists() else set()
    todo = [r for r in rows if r["src"] not in done]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    parser = nlp()
    with open(FEAT, "a") as f:
        for n in range(0, len(todo), 32):
            if time.monotonic() > deadline:
                break
            chunk = todo[n:n + 32]
            got = DF.features_many(parser, [(r["before"], Q.start_of(r["before"], r["after"]), r["word"], r["rep"])
                                            for r in chunk])
            for r, g in zip(chunk, got):
                f.write(json.dumps({"src": r["src"], **g}) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {n + len(chunk)}", flush=True)
    left = len({r["src"] for r in rows} - {r["src"] for r in read_jsonl(FEAT)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


# --- arm f ------------------------------------------------------------------------------

FRAME = ("dep_max_drop", "dep_top_drop", "dep_head_drop", "frame_slot_d")
LEX = ("sp_cplx", "sp_ppdb", "sp_miss", "wc_diff", "wc_miss", "spell")
SK = list(S.STAGE_KEYS)
TECH_ROUNDS = Q.TECH + ("r21",)
GENRE_ROUNDS = ("r21", "r22")
EXPLORE = REPORT / "r24-explore.json"


class FramePrior:
    """Smoothed Good rates of (target, replacement) lemma pairs in one dependency frame: the pair with
    one (relation, neighbour lemma) key of the target, e.g. (administer, give, child:advmod, orally).
    Each rate shrinks to the pair's own rate with FRAME_SMOOTH pseudo-counts; the feature is the most
    negative frame deviation, logit(frame rate) - logit(pair rate), over the swap's keys (0 if none
    has data). Learned from marks; never a veto."""

    def __init__(self, rows):
        self.c = {}
        for r in rows:
            for k in r["nb"]:
                x = self.c.setdefault((r["key"][0], r["key"][1], k[0], k[1]), [0, 0])
                x[0] += r["good"]
                x[1] += 1

    def worst(self, r, pp, own=None):
        out = 0.0
        for k in r["nb"]:
            g, n = self.c.get((r["key"][0], r["key"][1], k[0], k[1]), (0, 0))
            if own is not None:
                g, n = g - int(own), n - 1
            if n > 0:
                out = min(out, P12._logit((g + FRAME_SMOOTH * pp) / (n + FRAME_SMOOTH)) - P12._logit(pp))
        return out


FRAME_SMOOTH = 2.0


class ArmF:
    """Arm f: arm p's four features (logit arm-s score, sense cover, smoothed pair and target priors),
    round 21's two context features, the dependency-frame prior, and optionally the masked-LM frame
    features, the public pair-lexicon features and an unseen-pair indicator with interactions
    (logit arm-s score, SimplePPDB++ complexity, lexicon difference, ctx_cos). Standardized
    logistic, C = 0.1; priors leave the row out (a pair rated once is unseen in training)."""

    def __init__(self, rows, prior=True, frame=True, lex=True, inter=True):
        self.prior, self.frame, self.lex, self.inter = prior, frame, lex, inter
        y = [r["good"] for r in rows]
        self.pri = P12.Priors([r["key"] for r in rows], y)
        self.fp = FramePrior(rows)
        self.clf = S._fit(self.design(rows, loo=True), np.array(y))

    def design(self, rows, loo):
        X = []
        for r in rows:
            own = r["good"] if loo else None
            pp, pt, n = self.pri.rates(r["key"], own)
            ls = P12._logit(r["stack"])
            x = [ls, float(r["cover"]), P12._logit(pp), P12._logit(pt), *r["ctx"]]
            if self.prior:
                x.append(self.fp.worst(r, pp, own))
            if self.frame:
                x += [r["fr"][k] for k in FRAME]
            if self.lex:
                x += [r["lex"][k] for k in LEX]
            if self.inter:
                u = float(n <= 0)
                x += [u, u * ls, u * r["lex"]["sp_cplx"], u * r["lex"]["wc_diff"], u * r["x31"][SK.index("ctx_cos")]]
            X.append(x)
        return np.nan_to_num(np.array(X, float))

    def score(self, rows):
        return self.clf.predict_proba(self.design(rows, loo=False))[:, 1]


def _variant(**kw):
    return lambda rows: ArmF(rows, **kw)


NB = DIR / "pool-frame-keys.json"


def frame_keys(rows):
    """Dependency frame keys of every pool row (spaCy parse; cached in NB)."""
    from cleartext.features import nlp
    from cleartext.learned_words import depframe as DF

    if NB.exists():
        return json.load(open(NB))
    parser = nlp()
    out = {r["src"]: DF.frame_keys(parser(r["before"]), Q.start_of(r["before"], r["after"])) for r in rows}
    json.dump(out, open(NB, "w"))
    return out


def attach(rows, lexicon=None):
    """Frame keys, masked-LM frame features (from FEAT) and pair-lexicon features on pool rows."""
    from cleartext.learned_words.pairlex import PairLex

    lexicon = lexicon or PairLex()
    fr = {r["src"]: r for r in read_jsonl(FEAT)}
    nb = frame_keys(rows)
    for r in rows:
        r["fr"] = fr[r["src"]]
        r["nb"] = [tuple(k) for k in nb[r["src"]]]
        r["lex"] = lexicon.features(r["word"], r["rep"], r["key"])
    return rows


def _refs(rows):
    b = np.array([r["b"] for r in rows])
    rnd = np.array([r["round"] for r in rows])
    tech = b & np.array([r["round"] in Q.TECH or (r["round"] == "r21" and not r["genre"]) for r in rows])
    gen = b & np.array([bool(r.get("genre")) for r in rows])
    return tech, gen, rnd


def _well_rated(sub_rows, rows, idx):
    """Held-out rows whose pair has a training-fold prior >= 0.7 over >= 3 rated rows."""
    c = {}
    for r in sub_rows:
        x = c.setdefault(tuple(r["key"]), [0, 0])
        x[0] += r["good"]
        x[1] += 1
    out = []
    for i in idx:
        g, n = c.get(tuple(rows[i]["key"]), (0, 0))
        out.append(n >= 3 and g / n >= 0.7)
    return np.array(out, bool)


def nested_f(cls, rows, groups, seed, keep_tech, unseen_rule=None):
    """Passage-grouped nested CV: per outer split, the seen threshold keeps `keep_tech` of the training
    folds' technical arm-b rows by inner out-of-fold score; with `unseen_rule` (target precision),
    a separate threshold for pairs unseen in training is the lowest inner score whose kept unseen
    inner rows reach that precision (at least 15 kept), else the seen threshold."""
    tech, _, _ = _refs(rows)
    keyp = np.array(["|".join(r["key"]) for r in rows])
    y = np.array([r["good"] for r in rows])
    kp, seen, well = np.zeros(len(rows), bool), np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    score = np.zeros(len(rows))
    for tr, te in P12._folds(groups, seed):
        sub = [rows[i] for i in tr]
        inner = P14.oof(cls, sub, groups[tr], seed + 1)
        t = np.quantile(inner[tech[tr]], 1 - keep_tech)
        tu = t
        if unseen_rule is not None:
            # inner rows whose pair is unseen in their own inner training folds
            iu = np.zeros(len(tr), bool)
            for itr, ite in P12._folds(groups[tr], seed + 1):
                sp = set(keyp[tr][itr])
                iu[ite] = [k not in sp for k in keyp[tr][ite]]
            ref_u = iu & np.array([rows[i]["b"] for i in tr])
            o = np.argsort(-inner[ref_u])
            yy, ss = y[tr][ref_u][o], inner[ref_u][o]
            prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
            ok = [k for k in range(14, len(yy)) if prec[k] >= unseen_rule]
            tu = float(ss[ok[-1]]) if ok else max(t, float(ss[0]) + 1e-9)
        score[te] = cls(sub).score([rows[i] for i in te])
        sp = set(keyp[tr])
        seen[te] = [k in sp for k in keyp[te]]
        kp[te] = np.where(seen[te], score[te] >= t, score[te] >= tu)
        well[te] = _well_rated(sub, rows, te)
    return score, kp, seen, well


def _cvstats(kp, seen, well, marks, ref):
    m = Counter(marks[kp & ref])
    u = kp & ref & ~seen
    return {**P12._stats(kp, marks, ref), "unseen_rows": int((ref & ~seen).sum()), "unseen_kept": int(u.sum()),
            "unseen_good": int((marks[u] == "good").sum()),
            "frame_proxy_meaning_kept": int((marks[kp & ref & well] == "meaning").sum()),
            "frame_proxy_meaning_rows": int((marks[ref & well] == "meaning").sum()),
            "notsimpler": int(m["notsimpler"])}


def _mean(runs):
    return {f: float(np.mean([r[f] for r in runs])) for f in runs[0]}


def explore(argv):
    """Development diagnostics on the 1,454 rated rows (all already seen by earlier design steps).
    Nothing is kept; the CV gate and the fresh window decide."""
    from sklearn.metrics import roc_auc_score

    rows = attach(json.load(open(POOL)))
    marks = np.array([r["mark"] for r in rows])
    y = marks == "good"
    groups = np.array([r["group"] for r in rows])
    tech, gen, _ = _refs(rows)
    res = {"rows": len(rows), "tech_ref": int(tech.sum()), "genre_ref": int(gen.sum())}
    # single-feature AUCs: all rows, and inside well-rated pairs (leave-one-out prior >= 0.7 over >= 3)
    c = {}
    for r in rows:
        x = c.setdefault(tuple(r["key"]), [0, 0])
        x[0] += r["good"]
        x[1] += 1
    wr = np.array([(c[tuple(r["key"])][1] - 1) >= 3 and (c[tuple(r["key"])][0] - r["good"])
                   / (c[tuple(r["key"])][1] - 1) >= 0.7 for r in rows])
    single = {}
    for k in FRAME + ("dep_sum_drop", "dep_top_lp", "dep_n") + LEX:
        v = np.array([r["fr"][k] if k in r["fr"] else r["lex"][k] for r in rows])
        gm = marks != "notsimpler"
        single[k] = {"good_vs_meaning_all": float(roc_auc_score(y[gm], v[gm])),
                     "good_vs_meaning_well_rated": float(roc_auc_score(y[gm & wr], v[gm & wr])),
                     "good_vs_notsimpler_all": float(roc_auc_score(y[marks != "meaning"], v[marks != "meaning"]))}
    res["single_auc"] = single
    res["well_rated_rows"] = {"rows": int(wr.sum()), "meaning": int((marks[wr] == "meaning").sum())}
    variants = {"p": P12.ArmP, "q": Q.ArmQ, "prior": _variant(frame=False, lex=False, inter=False),
                "mlm": _variant(prior=False, lex=False, inter=False),
                "prior_mlm": _variant(lex=False, inter=False), "prior_lex": _variant(frame=False, inter=False),
                "prior_lex_inter": _variant(frame=False), "prior_lex_mlm": _variant(inter=False),
                "all": _variant()}
    cvr = {}
    for name, cls in variants.items():
        runs = {"tech": [], "genre": []}
        for k in range(5):
            _, kp, seen, well = nested_f(cls, rows, groups, P12.SEED + 10 * k, 0.50)
            runs["tech"].append(_cvstats(kp, seen, well, marks, tech))
            runs["genre"].append(_cvstats(kp, seen, well, marks, gen))
        cvr[name] = {part: _mean(v) for part, v in runs.items()}
        cvr[name]["good_by_repeat"] = [r["good_rate"] for r in runs["tech"]]
        print(name, json.dumps({p: {k: round(v, 3) for k, v in cvr[name][p].items()} for p in runs}), flush=True)
    for name, cls, rule in (("prior_lex_u90", _variant(frame=False, inter=False), 0.90),
                            ("prior_lex_u85", _variant(frame=False, inter=False), 0.85), ("p_u90", P12.ArmP, 0.90)):
        runs = {"tech": [], "genre": []}
        for k in range(5):
            _, kp, seen, well = nested_f(cls, rows, groups, P12.SEED + 10 * k, 0.50, unseen_rule=rule)
            runs["tech"].append(_cvstats(kp, seen, well, marks, tech))
            runs["genre"].append(_cvstats(kp, seen, well, marks, gen))
        cvr[name] = {part: _mean(v) for part, v in runs.items()}
        print(name, json.dumps(cvr[name]), flush=True)
    res["cv"] = cvr
    json.dump(res, open(EXPLORE, "w"), indent=1)
    print(json.dumps(res["single_auc"], indent=1))


# --- CV gate and pre-commitment --------------------------------------------------------

KEEP_TECH = 0.50  # arm f: share of technical arm-b rows kept (p50's rule), used on every passage
KEEP_GENRE = 0.40  # arm fg: share of genre arm-b rows kept (qg's rule), used on news and encyclopedic passages
REPEATS = 10
F_ARGS = {"prior": True, "frame": True, "lex": True, "inter": False}


def make_f(rows):
    """Arm f as pre-registered: frame prior, masked-LM frame features and pair lexicon; no interactions."""
    return ArmF(rows, **F_ARGS)


def _genre_runs(cls, rows, groups):
    """qg's genre-threshold CV (round 22): per outer split, the genre threshold keeps KEEP_GENRE of the
    training folds' genre arm-b rows by inner out-of-fold score; reported on held-out genre arm-b rows."""
    marks = np.array([r["mark"] for r in rows])
    keyp = np.array(["|".join(r["key"]) for r in rows])
    _, gen, _ = _refs(rows)
    runs = []
    for k in range(REPEATS):
        seed = P12.SEED + 10 * k
        kp, seen = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
        for tr, te in P12._folds(groups, seed):
            sub = [rows[i] for i in tr]
            inner = P14.oof(cls, sub, groups[tr], seed + 1)
            tg = np.quantile(inner[gen[tr]], 1 - KEEP_GENRE)
            kp[te] = cls(sub).score([rows[i] for i in te]) >= tg
            sp = set(keyp[tr])
            seen[te] = [x in sp for x in keyp[te]]
        runs.append(_cvstats(kp, seen, np.zeros(len(rows), bool), marks, gen))
    return _mean(runs)


def _gate_runs(cls, rows, groups):
    marks = np.array([r["mark"] for r in rows])
    tech, gen, _ = _refs(rows)
    runs = []
    for k in range(REPEATS):
        _, kp, seen, well = nested_f(cls, rows, groups, P12.SEED + 10 * k, KEEP_TECH)
        runs.append({"tech": _cvstats(kp, seen, well, marks, tech), "genre": _cvstats(kp, seen, well, marks, gen),
                     "both": _cvstats(kp, seen, well, marks, tech | gen)})
    return {part: _mean([r[part] for r in runs]) for part in runs[0]} | {
        "good_by_repeat": [r["tech"]["good_rate"] for r in runs]}


def train(argv):
    """The CV gate of ROUND24.md; if go, the final arm-f fit, both thresholds and the pre-commitment."""
    from cleartext.learned_words import pairlex as PL

    from . import learned_words_r9 as R

    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    rows = attach(json.load(open(POOL)))
    if len(rows) != POOL_ROWS:
        raise SystemExit(f"REFUSED: pool has {len(rows)} rows")
    groups = np.array([r["group"] for r in rows])
    tech, gen, _ = _refs(rows)
    res = {"rows": len(rows), "tech_ref": int(tech.sum()), "genre_ref": int(gen.sum()), "keep_tech": KEEP_TECH,
           "keep_genre": KEEP_GENRE, "repeats": REPEATS, "f_args": F_ARGS}
    res["r50"] = _gate_runs(P12.ArmP, rows, groups)
    res["f"] = _gate_runs(make_f, rows, groups)
    res["paired_gain"] = float(np.mean(np.array(res["f"]["good_by_repeat"]) - np.array(res["r50"]["good_by_repeat"])))
    res["fg_genre"] = _genre_runs(make_f, rows, groups)
    res["rg_genre"] = _genre_runs(P12.ArmP, rows, groups)
    r, f = res["r50"], res["f"]
    res["go_conditions"] = {
        "tech_good_ge_0.92": f["tech"]["good_rate"] >= 0.92,
        "tech_keep_ge_0.45": f["tech"]["keep_share"] >= 0.45,
        "paired_gain_ge_0.005": res["paired_gain"] >= 0.005,
        "frame_proxy_kept_le_0.85x_r50": f["tech"]["frame_proxy_meaning_kept"] <= 0.85 * r["tech"]["frame_proxy_meaning_kept"],
        "unseen_kept_ge_r50": f["both"]["unseen_kept"] >= r["both"]["unseen_kept"],
        "unseen_good_rate_ge_r50_minus_0.02": f["both"]["unseen_good"] / max(1, f["both"]["unseen_kept"])
        >= r["both"]["unseen_good"] / max(1, r["both"]["unseen_kept"]) - 0.02}
    res["go"] = bool(all(res["go_conditions"].values()))
    g = res["fg_genre"]
    res["fg_check"] = {"genre_good_ge_0.90": g["good_rate"] >= 0.90, "genre_keep_ge_0.35": g["keep_share"] >= 0.35}
    res["fg_go"] = bool(res["go"] and all(res["fg_check"].values()))
    res["pool_sha256"] = _sha(POOL)
    res["frame_features_sha256"] = _sha(FEAT)
    res["frame_keys_sha256"] = _sha(NB)
    res["source_sha256"] = {p: _sha(p) for p in SOURCES}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "source_sha256"}, indent=1))
    if not res["go"]:
        print("NO-GO: no fresh text is spent")
        return
    full = np.mean([P14.oof(make_f, rows, groups, P12.SEED + k) for k in range(REPEATS)], axis=0)
    t, tg = float(np.quantile(full[tech], 1 - KEEP_TECH)), float(np.quantile(full[gen], 1 - KEEP_GENRE))
    pickle.dump(make_f(rows), open(MODEL_F, "wb"))
    pre19, pre22 = json.load(open(P19.PRECOMMIT)), json.load(open(G.PRECOMMIT))
    json.dump({"thresholds": {"f": t, "fg": tg, "p50": pre19["thresholds"]["p50"],
                              "qg": pre22["thresholds"]["qg"], "qg_tech": pre22["thresholds"]["qg_tech"]},
               "keep_tech": KEEP_TECH, "keep_genre": KEEP_GENRE, "f_args": F_ARGS,
               "arms": [a for a in ARMS if a != "fg" or res["fg_go"]],
               "model_f_sha256": _sha(MODEL_F), "model_p50_sha256": _sha(P19.MODEL), "model_qg_sha256": _sha(G.MODEL_QG),
               "pool_sha256": _sha(POOL), "frame_features_sha256": _sha(FEAT), "frame_keys_sha256": _sha(NB),
               "pair_lexical_sha256": _sha(PL.LEXICAL), "wc_lexicon_sha256": _sha(PL.RAW_WC),
               "simpleppdbpp_raw_sha256": _sha(PL.RAW_PPDB),
               "sense_table_sha256": _sha(R.TABLE), "arm_s_model_sha256": _sha(S.MODEL),
               "genre_corpus_sha256": {n: _sha(GEN.GENRE / f"{n}.jsonl.gz") for n in EXT_NAMES.values()},
               "openfda_corpus_sha256": _sha(Q.R.CORPUS / "openfda.jsonl.gz"),
               "source_sha256": {p: _sha(p) for p in SOURCES}, "panel_seed": SEED,
               "per_source": {k: v[1] for k, v in PER_SOURCE.items()},
               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
    print(f"pre-committed thresholds f={t:.4f} fg={tg:.4f}")
    print("COMPLETE")


# --- fresh window ---------------------------------------------------------------------

def _docs(name):
    path = (GEN.GENRE if name in EXT_NAMES.values() else Q.R.CORPUS) / f"{name}.jsonl.gz"
    docs = {}
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        docs.setdefault(r["doc"], []).append(r["s"])
    return docs


def panel():
    """Round 22's panel rule (seed SEED) on the extended genre corpora (titles 220-520, never fetched
    before) and unused openFDA labels. Excludes every document of every earlier window (rounds 11-22,
    both Joshua windows) and any passage sharing a word 8-gram with any earlier panel or window,
    any rated development sentence or test-set v3's panel file (exclusion only)."""
    from cleartext.learned_words.rewrite import tokens

    from . import learned_words_r9 as R

    if PANEL.exists():
        raise SystemExit(f"REFUSED: {PANEL} exists")
    prev = []
    for p in (R.PANEL, Q.P12.PANEL, Q.P13.PANEL, Q.P17.PANEL, Q.P18.PANEL, Q.P19.PANEL, Q.P20.PANEL,
              Q.DIR / "passages.json", Q.JDIR / "passages.json", STATE / "joshua-r21b/passages.json", G.PANEL):
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
            out.append({"id": f"r24-{dom}-{n:02d}", "domain": dom, "kind": "standard", "source_doc": doc,
                        "corpus": name, "first_sentence": i, "original": text,
                        "sha256": hashlib.sha256(text.encode()).hexdigest()})
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


def load_models():
    pre = json.load(open(PRECOMMIT))
    from . import learned_words_r9 as R

    for path, k in ((MODEL_F, "model_f_sha256"), (P19.MODEL, "model_p50_sha256"), (G.MODEL_QG, "model_qg_sha256"),
                    (R.TABLE, "sense_table_sha256"), (S.MODEL, "arm_s_model_sha256")):
        if _sha(path) != pre[k]:
            raise SystemExit(f"REFUSED: {path.name} differs from the round-24 pre-commitment")
    from cleartext.learned_words import pairlex as PL

    for path, k in ((PL.LEXICAL, "pair_lexical_sha256"), (PL.RAW_WC, "wc_lexicon_sha256")):
        if _sha(path) != pre[k]:
            raise SystemExit(f"REFUSED: {path.name} differs from the round-24 pre-commitment")
    return {"f": pickle.load(open(MODEL_F, "rb")), "p50": pickle.load(open(P19.MODEL, "rb")),
            "qg": pickle.load(open(G.MODEL_QG, "rb"))}, pre


def apply(argv):
    """Round 22's apply on the round-24 window (checkpointed), plus the frame features of each
    resolved arm-b swap."""
    from cleartext.features import nlp
    from cleartext.learned_words import depframe as DF
    from cleartext.learned_words import senses as SN

    from . import learned_words_r9 as R
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
    done = {r["pid"] for r in read_jsonl(APPLY)} if APPLY.exists() else set()
    todo = [p for p in pan if p["id"] not in done and p["id"] not in skip]
    print(f"{len(done)} done, {len(todo)} to do, {len(skip)} skipped for rewrite-corpus overlap", flush=True)
    if todo:
        clf_s, _ = S.load_filter()
        sn = SN.load(R.TABLE)
        _, T, ids, _ = load()
        model = pickle.load(open(STATE / "model.pkl", "rb"))
        simpler = pickle.load(open(SIMPLER, "rb"))
        parser = nlp()
    with open(APPLY, "a") as f:
        for p in todo:
            if time.monotonic() > deadline:
                break
            variants, seen = R.word_swaps_two(p["original"], model, p["id"], sn)
            rec = {"pid": p["id"], "domain": p["domain"], "content_words": seen, "sentences": len(sent(p["original"]))}
            swaps = variants["all"]
            got, rows = [], []
            for x in swaps:
                r = S.swap_row(model, simpler, x)
                if r is not None:
                    got.append(x)
                    rows.append(r)
            if got:
                X = Q.P13._x31(T, ids, got, rows)
                fr = DF.features_many(parser, [(x["before"], r["start"], x["word"], x["rep"]) for x, r in zip(got, rows)])
                for x, r, xx, sc, ff in zip(got, rows, X, clf_s.predict_proba(X)[:, 1], fr):
                    x.update(margin=r["f"]["margin"], stack=float(sc), x31=[float(v) for v in xx],
                             lemma=r["lemma"], start=r["start"], fr=ff,
                             nb=DF.frame_keys(parser(x["before"]), r["start"]))
            rec["all"] = [x for x in swaps if "stack" in x]
            rec["unresolved_all"] = len(swaps) - len(rec["all"])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} all={len(rec['all'])}", flush=True)
    left = len({p["id"] for p in pan} - skip - {r["pid"] for r in read_jsonl(APPLY)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


def swap_rows(swaps, lexicon):
    """Arm-f input rows for applied swaps (no label)."""
    out = []
    for s in swaps:
        key = list(P12.pair_key(s["word"], s["rep"], s["pos"]))
        out.append({"key": key, "stack": s["stack"], "cover": s["cover"] or 0.0, "good": 0, "x31": s["x31"],
                    "ctx": Q.context(s["before"], s["after"], s["rep"]), "fr": s["fr"], "nb": [tuple(k) for k in s["nb"]],
                    "lex": lexicon.features(s["word"], s["rep"], key)})
    return out


def picks(rec, models, thr, choose, lexicon):
    """b: all candidates, margin. p50: round 19's frozen model at its threshold. qg: round 22's frozen
    model, its genre threshold on news and encyclopedic passages and its technical one on openFDA.
    f: arm f at its technical threshold on every passage. fg: arm f at its genre threshold on news and
    encyclopedic passages, its technical one on openFDA. Filters apply before the per-passage limits."""
    a = [s for s in rec["all"] if s["margin"] >= S.MIN_MARGIN]
    genre = rec["domain"] in GENRES
    if a:
        for s, x in zip(a, P12.score_p(models["p50"], a)):
            s["sc_p50"] = float(x)
        for s, x in zip(a, models["qg"].score(Q._rows_q(a))):
            s["sc_qg"] = float(x)
        rows = swap_rows(a, lexicon)
        for s, r, x in zip(a, rows, models["f"].score(rows)):
            s["sc_f"] = float(x)
            s["seen"] = bool(models["f"].pri.seen(r["key"]))
    return {"b": choose(a), "p50": choose([s for s in a if s["sc_p50"] >= thr["p50"]]),
            "qg": choose([s for s in a if s["sc_qg"] >= (thr["qg"] if genre else thr["qg_tech"])]),
            "f": choose([s for s in a if s["sc_f"] >= thr["f"]]),
            "fg": choose([s for s in a if s["sc_f"] >= (thr["fg"] if genre else thr["f"])])}


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
    rng = random.Random(1224)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r24") for x in items] + [dict(x, source="old", old_mark=x["mark"], shown=x["after"])
                                                     for x in old[:P12.CONSISTENCY_ITEMS]]
    RATING.mkdir(parents=True, exist_ok=True)
    for r in range(1, P12.RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r24" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    brief = (STATE / "r9/rater-brief.txt").read_text()
    for r in range(1, P12.RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1224:{r}").shuffle(order)
        path = RATING / f"blind-{r}.json"
        json.dump([{"id": ids[n], "before": x["before"], "after": x["shown"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(path, "w"), indent=1)
        n = len(base)
        b = brief.replace(str(STATE / "rating-r5/blind-mixed.json"), str(path)).replace("254 items", f"{n} items")
        b = b.replace("(all 254, ids m1..m254)", f"(all {n}, ids m1..m{n})")
        b = b.replace(str(STATE / "rating-r5/blind-mixed-marks.csv"), str(RATING / f"marks-{r}.csv"))
        b = b.replace("has 255 lines", f"has {n + 1} lines")
        (RATING / f"brief-{r}.txt").write_text(b)
    print(json.dumps({"r24_items": len(items), "items": len(base), "realized": sum(x["shown"] != x["after"] for x in items),
                      "per_arm": Counter(a for x in items for a in x["arms"]),
                      "per_domain": Counter(x["domain"] for x in items)}, indent=1))


def score(argv):
    from . import learned_words_r9 as R
    from .learned_words_study import wilson

    load_models()
    pool_rows = json.load(open(POOL))
    pool_pairs = {tuple(r["key"]) for r in pool_rows}
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
    by_dom = Counter()
    for r in recs:
        by_dom[r["domain"]] += r["sentences"]

    def pk(m):
        s = key[m["key"]]
        return tuple(P12.pair_key(s["word"], s["rep"], s["pos"]))

    arms = json.load(open(PRECOMMIT))["arms"]

    def arm_stats(marks):
        res = {}
        for arm in arms:
            got = [m for m in mp if m["source"] == "r24" and arm in m["arms"]]
            ms = [marks[m["id"]] for m in got]
            cc = Counter(ms)
            un = [marks[m["id"]] for m in got if pk(m) not in pool_pairs]
            fw = [marks[m["id"]] for m in got if pk(m) in well]
            res["arm_" + arm] = {"swaps": len(ms), **cc, "good_rate": cc["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(cc["good"], len(ms)), "meaning_rate": cc["meaning"] / max(1, len(ms)),
                                 "per_100_sentences": 100 * len(ms) / max(1, sentences),
                                 "distinct_pairs": len({pk(m) for m in got}),
                                 "unseen_pairs": {"swaps": len(un), "good": un.count("good"), "meaning": un.count("meaning")},
                                 "well_rated_pairs": {"swaps": len(fw), "meaning": fw.count("meaning")},
                                 "by_domain": {d: {"swaps": sum(key[m["key"]]["domain"] == d for m in got),
                                                   "good": sum(key[m["key"]]["domain"] == d and marks[m["id"]] == "good"
                                                               for m in got)} for d in PER_SOURCE}}
        return res

    out = {"items": len(ids), "raters": P12.RATERS, "passages": len(recs), "sentences": sentences,
           "sentences_by_domain": dict(by_dom), "majority": arm_stats(maj), "per_rater": [arm_stats(m) for m in raters]}
    M = out["majority"]
    out["yield_vs_b"] = {a: M["arm_" + a]["swaps"] / max(1, M["arm_b"]["swaps"]) for a in arms}
    v, p = M["arm_f"], M["arm_p50"]
    out["success_f"] = {"good_ge_0.90": v["good_rate"] >= 0.90, "meaning_le_0.08": v["meaning_rate"] <= 0.08,
                        "yield_ge_0.40": out["yield_vs_b"]["f"] >= 0.40}
    out["success"] = bool(all(out["success_f"].values()))
    out["secondary_f"] = {
        "frame_errors_le_p50": v["well_rated_pairs"]["meaning"] <= p["well_rated_pairs"]["meaning"],
        "unseen_swaps_ge_1.25x_p50": v["unseen_pairs"]["swaps"] >= 1.25 * p["unseen_pairs"]["swaps"],
        "unseen_good_ge_0.90": v["unseen_pairs"]["good"] >= 0.90 * max(1, v["unseen_pairs"]["swaps"])}
    if "fg" in arms:
        w = M["arm_fg"]
        out["success_fg"] = {"good_ge_0.90": w["good_rate"] >= 0.90, "meaning_le_0.08": w["meaning_rate"] <= 0.08,
                             "yield_ge_0.40": out["yield_vs_b"]["fg"] >= 0.40}
    r24 = [x["id"] for x in mp if x["source"] == "r24"]
    out["unanimous_share"] = sum(len({rt[i] for rt in raters}) == 1 for i in r24) / max(1, len(r24))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r24) / max(1, len(r24))
                                 for a in range(P12.RATERS) for c in range(a + 1, P12.RATERS)}
    old = [x for x in mp if x["source"] == "old"]
    out["consistency_with_old"] = [sum(x["old_mark"] == rt[x["id"]] for x in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_rater"}, indent=1))


def main(argv):
    cmd = {"fetch": fetch, "features": features, "explore": explore, "train": train, "apply": apply, "page": page,
           "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r24 features|explore|train|apply|page|score")
