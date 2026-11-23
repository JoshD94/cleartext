"""Round 7: what human editors did to a word when rewriting into plain language.

Each content word of a source sentence is KEPT, REPLACED (1:1 aligned with a
different lemma) or DROPPED in its aligned plain sentence. ``Table`` holds
those events as sorted integer-key counts, split by fold so a row's own
documents can be subtracted (cross-fitting). ``matrix`` turns (word,
replacement, neighbours) into features for a gradient-boosting filter.
"""

import difflib
import re
from functools import lru_cache

import numpy as np
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

WORD = re.compile(r"[a-z]+(?:['’-][a-z]+)*")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
KEPT, REPLACED, DROPPED = 0, 1, 2
WIKI, TECH = 0, 1
FOLDS = 5
BITS = 21
EVENT_MAX_ZIPF = 6.4  # function words are not events
TARGET_MAX_ZIPF = 5.6  # words the stage could plausibly target
MIN_JACCARD = 0.35


def tokens(text):
    return WORD.findall(text.lower().replace("’", "'"))


@lru_cache(maxsize=1000000)
def zipf(w):
    return zipf_frequency(w, "en")


@lru_cache(maxsize=1000000)
def lemma(w):
    for pos in (wn.VERB, wn.NOUN, wn.ADJ, wn.ADV):
        m = wn.morphy(w, pos)
        if m:
            return m
    return w


def content(w):
    return w.isalpha() and len(w) >= 3 and 1.0 <= zipf(w) <= EVENT_MAX_ZIPF


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def lemma_set(toks):
    return {lemma(t) for t in toks if content(t)}


def sentence_pairs(source, target):
    """Best plain sentence for each source sentence (document-level pairs)."""
    src = [(s, lemma_set(tokens(s))) for s in SENT.split(source)]
    tgt = [(t, lemma_set(tokens(t))) for t in SENT.split(target)]
    best = {}
    for t, lt in tgt:
        for i, (_, ls) in enumerate(src):
            j = jaccard(ls, lt)
            if j >= MIN_JACCARD and j > best.get(i, (0.0, None))[0]:
                best[i] = (j, t)
    return [(src[i][0], t) for i, (_, t) in sorted(best.items())]


def align(a, b):
    """[(index in a, status, replacement or None)] for content words of a."""
    la, lb = [lemma(x) for x in a], [lemma(x) for x in b]
    in_a, in_b = set(la), set(lb)
    out = []
    ops = difflib.SequenceMatcher(None, la, lb, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        for k, i in enumerate(range(i1, i2)):
            if not content(a[i]):
                continue
            if tag == "equal" or la[i] in in_b:
                out.append((i, KEPT, None))
                continue
            r = None
            if tag == "replace" and i2 - i1 == j2 - j1:
                c = b[j1 + k]
                if c.isalpha() and len(c) >= 2 and zipf(c) <= EVENT_MAX_ZIPF and lemma(c) not in in_a:
                    r = c
            out.append((i, REPLACED, r) if r else (i, DROPPED, None))
    return out


def _kw(w, dom, status):
    return (np.asarray(w, np.int64) << 3) | (dom << 2) | status


def _kp(w, r, dom):
    return (((np.asarray(w, np.int64) << BITS) | np.asarray(r, np.int64)) << 1) | dom


def _kc(a, b, kept):
    return (((np.asarray(a, np.int64) << BITS) | np.asarray(b, np.int64)) << 1) | kept


class Count:
    """Counts of int64 keys; `get` is vectorized."""

    def __init__(self, keys=None, k=None, n=None):
        if keys is not None:
            k, n = np.unique(keys, return_counts=True)
        self.k, self.n = k, n.astype(np.int32)

    def get(self, q):
        q = np.asarray(q, np.int64)
        if not len(self.k):
            return np.zeros(q.shape, np.int32)
        i = np.searchsorted(self.k, q).clip(0, len(self.k) - 1)
        return np.where(self.k[i] == q, self.n[i], 0)


class Table:
    NAMES = ("word", "pair", "prev", "next")

    def __init__(self, total, byfold):
        self.total, self.byfold = total, byfold

    @classmethod
    def build(cls, w, prev, nxt, status, r, fold, dom):
        kept = (status == KEPT).astype(np.int64)
        rep = status == REPLACED
        keys = {"word": (_kw(w, dom.astype(np.int64), status.astype(np.int64)), fold),
                "pair": (_kp(w[rep], r[rep], dom[rep].astype(np.int64)), fold[rep]),
                "prev": (_kc(prev, w, kept), fold), "next": (_kc(w, nxt, kept), fold)}
        total = {n: Count(k) for n, (k, _) in keys.items()}
        byfold = {n: Count(k * FOLDS + f) for n, (k, f) in keys.items()}
        return cls(total, byfold)

    def get(self, name, key, fold=None):
        t = self.total[name].get(key)
        if fold is None:
            return t
        fold = np.asarray(fold, np.int64)
        return np.where(fold >= 0, t - self.byfold[name].get(np.asarray(key, np.int64) * FOLDS + fold.clip(0)), t)

    def arrays(self):
        out = {}
        for part, d in (("t", self.total), ("f", self.byfold)):
            for n, c in d.items():
                out[f"{part}_{n}_k"], out[f"{part}_{n}_n"] = c.k, c.n
        return out

    @classmethod
    def from_arrays(cls, a):
        total = {n: Count(k=a[f"t_{n}_k"], n=a[f"t_{n}_n"]) for n in cls.NAMES}
        byfold = {n: Count(k=a[f"f_{n}_k"], n=a[f"f_{n}_n"]) for n in cls.NAMES}
        return cls(total, byfold)


@lru_cache(maxsize=500000)
def pair_static(w, r):
    zw, zr = zipf(w), zipf(r)
    lw, lr = lemma(w), lemma(r)
    syns = wn.synsets(lw)
    syn = any(lr == x.name().lower() for s in syns for x in s.lemmas())
    hyper = any(lr == x.name().lower() for s in syns[:3] for h in s.hypernyms() for x in h.lemmas())
    return (zw, zr, zr - zw, len(w) - len(r), float(syn), float(hyper), float(len(syns)))


STATIC = ("z_w", "z_r", "z_gain", "len_drop", "wn_syn", "wn_hyper", "w_senses")
SCOPED = ("n", "keep", "rep", "wr", "wr_of_rep", "wr_of_all", "direction", "r_n", "r_keep", "r_rep")
CONTEXT = ("prev_n", "prev_keep", "next_n", "next_keep", "ctx_keep_min", "ctx_lift")
KEYS = [f"{s}_{k}" for s in ("all", "tech") for k in SCOPED] + list(CONTEXT) + list(STATIC)


def _status(T, ids, fold, doms):
    return [sum(T.get("word", _kw(ids, d, st), fold) for d in doms) for st in (KEPT, REPLACED, DROPPED)]


def matrix(T, w, r, prev, nxt, fold, words, reps):
    """Feature matrix (columns = KEYS). fold = -1 means use all counts."""
    w, r, prev, nxt = (np.asarray(x, np.int64) for x in (w, r, prev, nxt))
    fold = np.asarray(fold, np.int64)
    out, base = {}, None
    for scope, doms in (("all", (WIKI, TECH)), ("tech", (TECH,))):
        k, rp, dp = _status(T, w, fold, doms)
        tot = k + rp + dp
        wr = sum(T.get("pair", _kp(w, r, d), fold) for d in doms)
        rw = sum(T.get("pair", _kp(r, w, d), fold) for d in doms)
        rk, rr, rd = _status(T, r, fold, doms)
        rt = rk + rr + rd
        keep = (k + 1) / (tot + 2)
        if scope == "all":
            base = keep
        vals = {"n": np.log1p(tot), "keep": keep, "rep": (rp + 0.5) / (tot + 1), "wr": np.log1p(wr),
                "wr_of_rep": wr / (rp + 1), "wr_of_all": wr / (tot + 1), "direction": np.log((1 + wr) / (1 + rw)),
                "r_n": np.log1p(rt), "r_keep": (rk + 1) / (rt + 2), "r_rep": (rr + 0.5) / (rt + 1)}
        out |= {f"{scope}_{n}": v for n, v in vals.items()}
    rates = []
    for name, a, b in (("prev", prev, w), ("next", w, nxt)):
        kept = T.get(name, _kc(a, b, 1), fold)
        n = kept + T.get(name, _kc(a, b, 0), fold)
        rate = (kept + 2 * base) / (n + 2)
        out[f"{name}_n"], out[f"{name}_keep"] = np.log1p(n), rate
        rates.append(rate)
    out["ctx_keep_min"] = np.minimum(*rates)
    out["ctx_lift"] = np.maximum(*rates) - base
    static = np.array([pair_static(a, b) for a, b in zip(words, reps)], float).reshape(len(words), len(STATIC))
    for i, name in enumerate(STATIC):
        out[name] = static[:, i]
    return np.column_stack([out[k] for k in KEYS])
