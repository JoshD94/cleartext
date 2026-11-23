"""Round 6, arm k: does the replacement fit this technical context?

Counts come from the round-6 technical reference corpus (RFC, Federal
Register and openFDA text, never a test-set-v3 document; see
evals/learned_words_corpus.py). For a swap target -> replacement at token i,
the features compare the two words in the same frames:

- keyness: corpus log-frequency per billion minus general Zipf frequency,
  for target, replacement and their difference
- for each frame (left word, right word, two left, two right, left+right):
  the log count with the target and with the replacement, and a
  PMI-style association (log count of the frame with the word, minus the
  log counts of the frame context and the word)
- summaries: the largest association drop target -> replacement, and how
  many frames the target is attested in while the replacement is not

``own`` subtracts one sentence's own n-grams, so a corpus sentence used as
a training row does not count itself.
"""

import math
import pickle
from collections import Counter
from functools import lru_cache

from wordfreq import zipf_frequency

FRAMES = ("L1", "R1", "L2", "R2", "M")


@lru_cache(1)
def load_counts(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def own_ngrams(tokens):
    return Counter(tokens), Counter(zip(tokens, tokens[1:])), Counter(zip(tokens, tokens[1:], tokens[2:]))


class Counts:
    def __init__(self, c, own=None):
        self.c, self.own = c, own or (Counter(), Counter(), Counter())
        self.n = c["tokens"]

    def uni(self, w):
        return max(0, self.c["uni"].get(w, 0) - self.own[0].get(w, 0))

    def bi(self, a, b):
        return max(0, self.c["bi"].get((a, b), 0) - self.own[1].get((a, b), 0))

    def tri(self, a, b, c):
        return max(0, self.c["tri"].get((a, b, c), 0) - self.own[2].get((a, b, c), 0))


def _frame_count(C, frame, toks, i, w):
    get = lambda k: toks[k] if 0 <= k < len(toks) else None  # noqa: E731
    l1, l2, r1, r2 = get(i - 1), get(i - 2), get(i + 1), get(i + 2)
    if frame == "L1":
        return (C.bi(l1, w), C.uni(l1)) if l1 else None
    if frame == "R1":
        return (C.bi(w, r1), C.uni(r1)) if r1 else None
    if frame == "L2":
        return (C.tri(l2, l1, w), C.bi(l2, l1)) if l1 and l2 else None
    if frame == "R2":
        return (C.tri(w, r1, r2), C.bi(r1, r2)) if r1 and r2 else None
    if frame == "M":
        return (C.tri(l1, w, r1), C.bi(l1, r1) + 1) if l1 and r1 else None
    raise ValueError(frame)


def features(C, toks, i, rep):
    """toks: lowercased word tokens; toks[i] is the target."""
    t, r = toks[i], rep.lower()
    f = {}
    for name, w in (("t", t), ("r", r)):
        per_billion = (C.uni(w) + 0.5) / C.n * 1e9
        f[f"key_{name}"] = math.log10(per_billion) - zipf_frequency(w, "en")
        f[f"cnt_{name}"] = math.log1p(C.uni(w))
    f["key_d"] = f["key_t"] - f["key_r"]
    drops, orphan = [], 0
    for fr in FRAMES:
        ct = _frame_count(C, fr, toks, i, t)
        cr = _frame_count(C, fr, toks, i, r)
        if ct is None:
            for k in ("ct", "cr", "at", "ar", "drop"):
                f[f"{fr}_{k}"] = -1.0 if k in ("ct", "cr") else 0.0
            continue
        (nt, ctx), (nr, _) = ct, cr
        at = math.log(nt + 0.5) - math.log(ctx + 1) - math.log((C.uni(t) + 1) / C.n)
        ar = math.log(nr + 0.5) - math.log(ctx + 1) - math.log((C.uni(r) + 1) / C.n)
        f[f"{fr}_ct"], f[f"{fr}_cr"] = math.log1p(nt), math.log1p(nr)
        f[f"{fr}_at"], f[f"{fr}_ar"] = at, ar
        f[f"{fr}_drop"] = at - ar
        drops.append(at - ar)
        orphan += int(nt >= 3 and nr == 0)
    f["max_drop"] = max(drops) if drops else 0.0
    f["mean_drop"] = sum(drops) / len(drops) if drops else 0.0
    f["orphan_frames"] = float(orphan)
    return f


KEYS = sorted(features(Counts({"tokens": 1, "uni": {}, "bi": {}, "tri": {}}), ["a", "b", "c", "d", "e"], 2, "x"))


def vector(C, toks, i, rep):
    f = features(C, toks, i, rep)
    return [f[k] for k in KEYS]


def save(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
