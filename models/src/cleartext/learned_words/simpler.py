"""Is the replacement actually simpler? A learned word-difficulty margin.

Round 2 (November 19) showed the swaps that fail most often as "not simpler"
are same-difficulty synonyms (assessing -> evaluating, Furthermore ->
Moreover). The ranker learned substitutability from gold lists, which reward
simplicity only implicitly. This model scores word difficulty directly.

``WordDifficulty``: two gradient-boosting regressors on word-type features
(frequency of form and lemma, length, syllables, wordpieces, WordNet senses
and lexicographer file, and a PCA of BERT's static input embedding):

- CompLex 2.0 single words (CC BY 4.0): mean 5-point complexity rating per
  word, scaled 0-1, averaged over its contexts (bible, biomed, europarl).
- CWI 2018 English (news, WikiNews, Wikipedia): share of 20 annotators who
  marked the word complex, averaged over its contexts.

Each regressor's output is z-scored with its training targets' mean and
standard deviation; difficulty is the mean of the two. The swap margin is
difficulty(target) - difficulty(replacement), in those z units.

Training uses CompLex train and CWI Train; the threshold rule reads only
validation words (CompLex trial, CWI Dev) never seen in training; test words
(CompLex test, CWI Test) are read once after the threshold is set. Nothing
here reads Joshua's marks or any round's rated swaps.
"""

import csv
from collections import defaultdict
from functools import lru_cache

import numpy as np
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

from ..data import RAW as V7_RAW

PCA_DIM = 16
# "Clearly simpler" in each source's own units: 0.10 of the 0-1 CompLex scale
# (0.4 of a Likert step) or 2 of CWI's 20 annotators.
DELTA = {"complex": 0.10, "cwi": 0.10}
TARGET_PRECISION = 0.80


def _word_ok(w):
    return w.isalpha() and len(w) >= 2


def complex_words(split):
    """{word: mean complexity} for CompLex single-word targets in one split."""
    acc = defaultdict(list)
    with open(V7_RAW / f"complex_{split}.tsv", newline="", encoding="utf8") as f:
        rows = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(rows)
        for r in rows:
            if len(r) < 5:
                continue
            w = r[3].strip().lower()
            if _word_ok(w):
                acc[w].append(float(r[4]))
    return {w: float(np.mean(v)) for w, v in acc.items()}


def cwi_words(split):
    """{word: mean share of annotators marking it complex} for CWI 2018 single words."""
    acc = defaultdict(list)
    with open(V7_RAW / f"cwi_english_{split}.tsv", newline="", encoding="utf8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) < 11:
                continue
            w = r[4].strip().lower()
            if " " in w or not _word_ok(w):
                continue
            acc[w].append(float(r[10]))
    return {w: float(np.mean(v)) for w, v in acc.items()}


def splits():
    """{source: {"train"|"val"|"test": {word: score}}}; val/test hold only unseen words."""
    out = {"complex": {"train": complex_words("train"), "val": complex_words("trial"),
                       "test": complex_words("test")},
           "cwi": {"train": cwi_words("Train"), "val": cwi_words("Dev"), "test": cwi_words("Test")}}
    for src in out:
        seen = set(out[src]["train"])
        out[src]["val"] = {w: s for w, s in out[src]["val"].items() if w not in seen}
        seen |= set(out[src]["val"])
        out[src]["test"] = {w: s for w, s in out[src]["test"].items() if w not in seen}
    return out


@lru_cache(200000)
def lemma_zipf(word):
    w = word.lower()
    lemmas = {wn.morphy(w, p) for p in "nvar"} - {None}
    return max([zipf_frequency(x, "en") for x in lemmas] or [zipf_frequency(w, "en")])


def form_features(words):
    from .eligibility import type_features

    out = []
    for w in words:
        f = dict(type_features(w))
        f["lemma_zipf"] = lemma_zipf(w)
        f["min_zipf"] = min(f["zipf"], f["lemma_zipf"])
        out.append(f)
    return out


class WordDifficulty:
    def __init__(self, seed=4701):
        self.seed = seed

    def _X(self, words):
        from .eligibility import static_embedding

        base = form_features(words)
        emb = self.pca.transform(static_embedding(words))
        return np.hstack([np.array([[d[k] for k in self.keys] for d in base], float), emb])

    def fit(self, train):
        """train: {source: {word: score}}."""
        from sklearn.decomposition import PCA
        from sklearn.ensemble import HistGradientBoostingRegressor

        from .eligibility import static_embedding

        allw = sorted({w for d in train.values() for w in d})
        self.keys = sorted(form_features(["example"])[0])
        self.pca = PCA(PCA_DIM, random_state=self.seed).fit(static_embedding(allw))
        self.models, self.norm = {}, {}
        for src, d in sorted(train.items()):
            words = sorted(d)
            y = np.array([d[w] for w in words])
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, min_samples_leaf=20,
                                              random_state=self.seed)
            self.models[src] = m.fit(self._X(words), y)
            self.norm[src] = (float(y.mean()), float(y.std()))
        return self

    def raw(self, words, src):
        return self.models[src].predict(self._X(list(words)))

    def difficulty(self, words):
        words = [w.lower() for w in words]
        if not words:
            return np.array([])
        X = self._X(words)
        z = [(self.models[s].predict(X) - mu) / sd for s, (mu, sd) in sorted(self.norm.items())]
        return np.mean(z, axis=0)

    def margin(self, targets, replacements):
        """difficulty(target) - difficulty(replacement), z units; > 0 means simpler."""
        d = self.difficulty(list(targets) + list(replacements))
        n = len(targets)
        return d[:n] - d[n:]


def pairs(scores, pred, delta, max_pairs=200000, seed=4701):
    """All (or a seeded sample of) ordered word pairs a -> b: predicted margin
    pred[a] - pred[b], label = observed score[a] - score[b] >= delta."""
    words = sorted(scores)
    rng = np.random.RandomState(seed)
    n = len(words)
    i = rng.randint(0, n, max_pairs)
    j = rng.randint(0, n, max_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    s = np.array([scores[w] for w in words])
    p = np.array([pred[w] for w in words])
    return p[i] - p[j], (s[i] - s[j]) >= delta, s[i] - s[j]


def choose_margin(margins, labels, target=TARGET_PRECISION, grid=None):
    """Smallest margin whose pairs at or above it are "clearly simpler" with
    precision >= target (pre-registered rule)."""
    grid = np.round(np.arange(0.0, 3.01, 0.05), 2) if grid is None else grid
    table = []
    for m in grid:
        sel = margins >= m
        prec = float(labels[sel].mean()) if sel.any() else None
        table.append({"margin": float(m), "pairs": int(sel.sum()), "precision": prec})
    ok = [r for r in table if r["precision"] is not None and r["precision"] >= target and r["pairs"] >= 100]
    return (ok[0]["margin"] if ok else None), table
