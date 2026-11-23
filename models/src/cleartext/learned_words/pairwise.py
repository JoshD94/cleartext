"""Round 6, arm p: is the replacement simpler than the target? A pairwise model.

The round-3 margin, difficulty(target) - difficulty(replacement), subtracts
two per-word regressors. It cannot see a pair's relation: a same-family
variant (intention -> intent, upward -> upwards), a spelling variant, or two
words of near-equal frequency. This classifier scores the pair directly.

Training pairs (public data only; nothing from any rated round):
- BenchLS-train (NNSeval instances removed): each ordered pair of an
  instance's substitutes, labelled 1 when the second word is ranked strictly
  simpler. Ties and reversed pairs are 0.
- BenchLS-train target -> rank-1 substitute (1), and the reverse (0).
- plainlanguage.gov single-word hard -> plain pairs (1), and the reverse (0).

Features: both words' type features (frequency, lemma frequency, length,
syllables, word pieces, WordNet senses), the round-3 regressors' raw
difficulty for each word, their differences, and pair relations (edit
distance, shared prefix, same WordNet lemma or derivational family, static
BERT cosine).
"""

import numpy as np
from nltk.corpus import wordnet as wn

BASE = ("zipf", "lemma_zipf", "min_zipf", "len", "syll", "pieces", "senses_n", "senses_v", "senses_a",
        "senses_r")


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _family(a, b):
    """Same WordNet lemma, or linked by a derivational relation."""
    la = {wn.morphy(a, p) for p in "nvar"} - {None} | {a}
    lb = {wn.morphy(b, p) for p in "nvar"} - {None} | {b}
    if la & lb:
        return 1.0
    for s in wn.synsets(a):
        for lem in s.lemmas():
            if lem.name().lower() in la and any(d.name().lower() in lb for d in lem.derivationally_related_forms()):
                return 1.0
    return 0.0


def pair_rows(pairs, difficulty):
    """pairs: [(target, replacement)] -> feature matrix, keys."""
    from .eligibility import static_embedding
    from .simpler import form_features

    words = sorted({w for p in pairs for w in p})
    form = dict(zip(words, form_features(words)))
    emb = dict(zip(words, static_embedding(words)))
    raw = {src: dict(zip(words, difficulty.raw(words, src))) for src in sorted(difficulty.models)}
    rows = []
    for a, b in pairs:
        fa, fb = form[a], form[b]
        r = {}
        for k in BASE:
            r[f"t_{k}"], r[f"r_{k}"], r[f"d_{k}"] = fa[k], fb[k], fa[k] - fb[k]
        for src, d in raw.items():
            r[f"t_{src}"], r[f"r_{src}"], r[f"d_{src}"] = d[a], d[b], d[a] - d[b]
        ea, eb = emb[a], emb[b]
        r["cos"] = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
        e = _edit(a, b)
        r["edit"], r["edit_rel"] = e, e / max(len(a), len(b))
        pre = 0
        while pre < min(len(a), len(b)) and a[pre] == b[pre]:
            pre += 1
        r["prefix"], r["prefix_rel"] = pre, pre / min(len(a), len(b))
        r["family"] = _family(a, b)
        rows.append(r)
    keys = sorted(rows[0]) if rows else []
    return np.array([[r[k] for k in keys] for r in rows], float), keys


class PairSimpler:
    def __init__(self, seed=4701):
        self.seed = seed

    def fit(self, pairs, y, difficulty):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.difficulty = difficulty
        X, self.keys = pair_rows(pairs, difficulty)
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                    l2_regularization=1.0, min_samples_leaf=20,
                                                    random_state=self.seed).fit(X, y)
        return self

    def prob(self, pairs):
        pairs = [(a.lower(), b.lower()) for a, b in pairs]
        if not pairs:
            return np.array([])
        X, keys = pair_rows(pairs, self.difficulty)
        assert keys == self.keys
        return self.model.predict_proba(X)[:, 1]
