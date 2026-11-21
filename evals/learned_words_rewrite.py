"""Round 7, arm w: learn the swap decision from real plain-language rewrites (see ROUND7.md).

  learned-words rewrite build    alignment events from the rewrite corpora
  learned-words rewrite train    cross-fitted model, threshold, rewrite-precommit.json
  learned-words rewrite replay   go/no-go on the round 3-5 replay set (after the pre-commit is in Git)

Raw text and events stay in the state directory (rewrite/).
"""

import hashlib
import json
import pickle
import random
import time
from array import array
from collections import Counter, defaultdict

import numpy as np

from .learned_words import REPORT, STATE, all_instances
from .learned_words_contextfit import candidates
from .learned_words_corpus import V3_STATE, _excluded_text

DIR = STATE / "rewrite"
RAW = DIR / "raw"
EVENTS = DIR / "events.npz"
VOCAB = DIR / "vocab.pkl"
TABLE = DIR / "table.npz"
MODEL = DIR / "model.pkl"
PRECOMMIT = REPORT / "rewrite-precommit.json"
SEED = 20261123
KEEP_POSITIVES = 0.70
MAX_ROWS = 300000
CAP_PAIR, CAP_WORD, CAP_KEEP, CAP_WRONG = 30, 200, 20, 10
FILES = {"cochrane-train.json": "4b52d2e8a09aecb0f9d1fe8894927e49ca019e8cbd588dfa762ef8daab93e119",
         "cochrane-validation.json": "18c883a77ff20f718b71c05251146d9648022f277cd4dc345b43820cd6ba2d5f",
         "wiki-test_contract.parquet": "1bbeb032a5d694b7d480d5abb38bee593b9bd03a445712c15b2ab0a5b4229e7c",
         "wiki-train.parquet": "ac2e3810ba66ed24f4c00e35cc9f6019ed85f135f0bb69382a7fd6709c346865",
         "wiki-validation.parquet": "ebf89244687b51a15aef34a35e72e058a4faca00273a4131c598ac5813834c38"}
SOURCES = ["models/src/cleartext/learned_words/rewrite.py", "evals/learned_words_rewrite.py"]


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fold_of(doc):
    from cleartext.learned_words.rewrite import FOLDS

    return int(hashlib.sha256(doc.encode()).hexdigest(), 16) % FOLDS


def exclusions():
    """(8-grams, exact short sentences) of everything rewrite data must not contain."""
    from cleartext.learned_words.rewrite import tokens

    grams, _, _, _ = _excluded_text()
    grams = set(grams)
    texts = [p["original"] for p in json.load(open(V3_STATE / "panel/all.json"))]
    onestop = json.load(open(V3_STATE / "panel/onestop-dev.json"))
    texts += [p["original"] for p in (onestop if isinstance(onestop, list) else onestop["passages"])]
    for name in ("rating/key.json", "rating-r3/key.json", "rating-r4/key.json", "rating-r5/key.json"):
        for s in json.load(open(STATE / name))["swaps"]:
            texts += [s["before"], s["after"]]
    for s in json.load(open(STATE / "replay/items.json")):
        texts += [s["before"], s["after"]]
    texts += [i["text"] for i in all_instances() if i["dataset"] in ("joshua60", "tsar_test", "nnseval", "swords_test")]
    short = set()
    for x in texts:
        t = tokens(x)
        grams.update(tuple(t[i:i + 8]) for i in range(len(t) - 7))
        if len(t) < 8:
            short.add(tuple(t))
    return grams, short


def blocked(toks, grams, short):
    if len(toks) < 8:
        return tuple(toks) in short
    return any(tuple(toks[i:i + 8]) in grams for i in range(len(toks) - 7))


def pairs():
    """(doc id, domain, source, target) sentence pairs."""
    import pyarrow.parquet as pq

    from cleartext.learned_words.rewrite import TECH, WIKI, sentence_pairs

    for split in ("train", "validation"):
        for x in json.load(open(RAW / f"cochrane-{split}.json")):
            for a, b in sentence_pairs(x["source"], x["target"]):
                yield x["doi"], TECH, a, b
    for name, dom in (("wiki-test_contract", TECH), ("wiki-validation", WIKI), ("wiki-train", WIKI)):
        table = pq.read_table(RAW / f"{name}.parquet", columns=["source", "target"])
        for a, b in zip(table["source"].to_pylist(), table["target"].to_pylist()):
            yield a, dom, a, b


def build(argv):
    from cleartext.learned_words.rewrite import BITS, MIN_JACCARD, align, jaccard, lemma_set, tokens

    for f, h in FILES.items():
        if _sha(RAW / f) != h:
            raise SystemExit(f"REFUSED: {f} hash differs")
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    grams, short = exclusions()
    print(f"excluded 8-grams {len(grams)}, short sentences {len(short)}", flush=True)
    vocab = {"<s>": 0}

    def vid(x):
        return vocab.setdefault(x, len(vocab))

    cols = {k: array("i") for k in ("w", "prev", "next", "r")}
    small = {k: array("b") for k in ("status", "fold", "dom")}
    stats = Counter()
    start = time.monotonic()
    for doc, dom, a, b in pairs():
        stats["pairs_seen"] += 1
        if limit and stats["pairs_seen"] > limit:
            break
        ta, tb = tokens(a), tokens(b)
        if not (5 <= len(ta) <= 70 and 3 <= len(tb) <= 90) or ta == tb:
            stats["dropped_shape"] += 1
            continue
        if jaccard(lemma_set(ta), lemma_set(tb)) < MIN_JACCARD:
            stats["dropped_overlap"] += 1
            continue
        if blocked(ta, grams, short) or blocked(tb, grams, short):
            stats["dropped_excluded"] += 1
            continue
        f = fold_of(doc)
        stats[f"pairs_used_dom{dom}"] += 1
        for i, status, r in align(ta, tb):
            cols["w"].append(vid(ta[i]))
            cols["prev"].append(vid(ta[i - 1]) if i else 0)
            cols["next"].append(vid(ta[i + 1]) if i + 1 < len(ta) else 0)
            cols["r"].append(vid(r) if r else 0)
            small["status"].append(status)
            small["fold"].append(f)
            small["dom"].append(dom)
            stats[f"status{status}_dom{dom}"] += 1
        if stats["pairs_seen"] % 50000 == 0:
            print(f"{time.strftime('%H:%M:%S')} {stats['pairs_seen']} pairs, {len(cols['w'])} events", flush=True)
    if len(vocab) >= 2 ** BITS:
        raise SystemExit("REFUSED: vocabulary exceeds key width")
    res = {**stats, "events": len(cols["w"]), "vocab": len(vocab), "seconds": round(time.monotonic() - start)}
    if limit:
        print(json.dumps(res, indent=1))
        print("LIMITED RUN: nothing written")
        return
    np.savez(EVENTS, **{k: np.frombuffer(v, np.int32) for k, v in cols.items()},
             **{k: np.frombuffer(v, np.int8) for k, v in small.items()})
    pickle.dump(list(vocab), open(VOCAB, "wb"))
    res["events_sha256"] = _sha(EVENTS)
    json.dump(res, open(REPORT / "rewrite-data.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("COMPLETE")


def sample(ev, words):
    """Training rows: human replacements (1), kept words and wrong words (0)."""
    from wordfreq import zipf_frequency

    from cleartext.learned_words import data as D
    from cleartext.learned_words.rewrite import KEPT, REPLACED, TARGET_MAX_ZIPF

    rng = random.Random(SEED)
    plain = D.plain_pairs()
    ids = {w: i for i, w in enumerate(words)}
    z = np.array([zipf_frequency(w, "en") for w in words])
    w, r, status = ev["w"], ev["r"], ev["status"]
    ok = (z[w] <= TARGET_MAX_ZIPF) & (w > 0)
    order = np.random.default_rng(SEED).permutation(np.flatnonzero(ok & ((status == KEPT) | (status == REPLACED))))
    cache, per_pair, per_word, keep_n, wrong_n = {}, Counter(), Counter(), Counter(), Counter()
    rows = []

    def cands(x):
        if x not in cache:
            cache[x] = candidates(words[x], plain)
        return cache[x]

    for e in order.tolist():
        x, d = int(w[e]), int(ev["dom"][e])
        if status[e] == REPLACED:
            y = int(r[e])
            if z[y] > z[x] and per_pair[(x, y)] < CAP_PAIR and per_word[(x, d)] < CAP_WORD:
                per_pair[(x, y)] += 1
                per_word[(x, d)] += 1
                rows.append((e, x, y, words[y], 1, "human"))
            if wrong_n[(x, d)] < CAP_WRONG:
                other = [c for c in cands(x) if c != words[y]]
                if other:
                    c = rng.choice(other)
                    wrong_n[(x, d)] += 1
                    rows.append((e, x, ids.get(c, 0), c, 0, "wrong"))
        elif keep_n[(x, d)] < CAP_KEEP and cands(x):
            c = rng.choice(cache[x])
            keep_n[(x, d)] += 1
            rows.append((e, x, ids.get(c, 0), c, 0, "keep"))
    rng.shuffle(rows)
    return rows[:MAX_ROWS]


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, min_samples_leaf=40, random_state=4701)


def train(argv):
    from sklearn.metrics import roc_auc_score

    from cleartext.learned_words.rewrite import FOLDS, KEYS, TECH, Table, matrix

    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    ev = dict(np.load(EVENTS))
    words = pickle.load(open(VOCAB, "rb"))
    T = Table.build(ev["w"], ev["prev"], ev["next"], ev["status"], ev["r"], ev["fold"], ev["dom"])
    print("table built", flush=True)
    rows = sample(ev, words)
    e = np.array([x[0] for x in rows])
    y = np.array([x[4] for x in rows])
    kinds = np.array([x[5] for x in rows])
    fold, dom = ev["fold"][e].astype(np.int64), ev["dom"][e].astype(np.int64)
    X = matrix(T, [x[1] for x in rows], [x[2] for x in rows], ev["prev"][e], ev["next"][e], fold,
               [words[x[1]] for x in rows], [x[3] for x in rows])
    print(f"rows {len(y)} by kind {Counter(kinds.tolist())} by domain {Counter(dom.tolist())}", flush=True)
    sw = np.zeros(len(y))
    for d in (0, 1):
        for c in (0, 1):
            m = (dom == d) & (y == c)
            sw[m] = len(y) / (4 * max(1, m.sum()))
    oof = np.zeros(len(y))
    for f in range(FOLDS):
        te = fold == f
        oof[te] = _model().fit(X[~te], y[~te], sample_weight=sw[~te]).predict_proba(X[te])[:, 1]
        print(f"fold {f} done", flush=True)
    tech_pos = (dom == TECH) & (y == 1)
    t = float(np.quantile(oof[tech_pos], 1 - KEEP_POSITIVES))
    res = {"rule": f"keep {KEEP_POSITIVES:.0%} of out-of-fold technical human replacements", "threshold": t,
           "rows": len(y), "by_kind": dict(Counter(kinds.tolist())), "technical_rows": int((dom == TECH).sum()),
           "oof_auc": {"all": float(roc_auc_score(y, oof, sample_weight=sw)),
                       "technical": float(roc_auc_score(y[dom == TECH], oof[dom == TECH])),
                       "general": float(roc_auc_score(y[dom != TECH], oof[dom != TECH])),
                       "vs_keep": float(roc_auc_score(y[kinds != "wrong"], oof[kinds != "wrong"])),
                       "vs_wrong": float(roc_auc_score(y[kinds != "keep"], oof[kinds != "keep"]))},
           "negatives_below_threshold": {k: float((oof[kinds == k] < t).mean()) for k in ("keep", "wrong")},
           "features": KEYS}
    model = _model().fit(X, y, sample_weight=sw)
    pickle.dump(model, open(MODEL, "wb"))
    np.savez(TABLE, **T.arrays())
    res |= {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "model_sha256": _sha(MODEL), "table_sha256": _sha(TABLE),
            "events_sha256": _sha(EVENTS), "source_sha256": {p: _sha(p) for p in SOURCES}}
    json.dump(res, open(PRECOMMIT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=1))
    print("COMPLETE")


def load():
    from cleartext.learned_words.rewrite import Table

    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL) != pre["model_sha256"] or _sha(TABLE) != pre["table_sha256"]:
        raise SystemExit("REFUSED: rewrite model or table differ from the pre-commitment")
    words = pickle.load(open(VOCAB, "rb"))
    return pickle.load(open(MODEL, "rb")), Table.from_arrays(np.load(TABLE)), {w: i for i, w in enumerate(words)}, pre["threshold"]


def score(model, T, ids, swaps):
    """swaps: dicts with before, word, rep (and after) -> P(an editor would make this swap)."""
    from cleartext.learned_words.rewrite import matrix, tokens

    W, R, P, N, ws, rs = [], [], [], [], [], []
    for s in swaps:
        toks = tokens(s["before"])
        w = tokens(s["word"])[0]
        r = tokens(s["rep"])[0]
        idx = [k for k, x in enumerate(toks) if x == w]
        k = idx[0] if idx else 0
        after = tokens(s.get("after") or "")
        for j in idx:
            if j < len(after) and after[j] == r:
                k = j
                break
        W.append(ids.get(w, 0))
        R.append(ids.get(r, 0))
        P.append(ids.get(toks[k - 1], 0) if k else 0)
        N.append(ids.get(toks[k + 1], 0) if k + 1 < len(toks) else 0)
        ws.append(w)
        rs.append(r)
    X = matrix(T, W, R, P, N, np.full(len(W), -1), ws, rs)
    return model.predict_proba(X)[:, 1]


def replay(argv):
    from sklearn.metrics import roc_auc_score

    model, T, ids, t = load()
    items = [i for i in json.load(open(STATE / "replay/items.json")) if "b" in i["arms"]]
    p = score(model, T, ids, items)
    kept = [i for i, x in zip(items, p) if x >= t]
    base, k = Counter(i["mark"] for i in items), Counter(i["mark"] for i in kept)
    errors = base["notsimpler"] + base["meaning"]
    res = {"threshold": t, "baseline": dict(base), "kept": dict(k), "n_base": len(items), "n_kept": len(kept),
           "good_rate_base": base["good"] / len(items), "good_rate_kept": k["good"] / max(1, len(kept)),
           "keep_share": len(kept) / len(items),
           "errors_removed_share": 1 - (k["notsimpler"] + k["meaning"]) / errors,
           "auc_good_vs_error": float(roc_auc_score([i["mark"] == "good" for i in items], p))}
    res["go"] = bool(res["good_rate_kept"] >= res["good_rate_base"] + 0.05 and res["keep_share"] >= 0.60
                     and res["errors_removed_share"] >= 1 / 3)
    json.dump(res, open(REPORT / "rewrite-replay.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("COMPLETE")


def main(argv):
    cmd = {"build": build, "train": train, "replay": replay}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words rewrite build|train|replay")
