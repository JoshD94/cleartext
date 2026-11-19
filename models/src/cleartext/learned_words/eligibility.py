"""Stage 1: which words are hard (CWI 2018) and safe to replace (not a term).

``ComplexWord``: gradient boosting on CWI 2018 English single-word targets
(binary label: at least one of 20 annotators marked the word complex), with
frequency, length, syllables, wordpieces, WordNet sense counts, part of
speech and neighbouring-word frequency. Threshold: best F1 on CWI Dev.

``TermWord``: a word-type classifier. Positives are single-word entries of
three public technical glossaries (NIST CSRC, US Courts, MeSH 2026);
negatives are words with attested simpler substitutes (plainlanguage.gov
"don't say" words, BenchLS/TSAR-trial targets, SWORDS-dev targets with an
accepted substitute). Words in both are negatives. Features: frequency,
shape, WordNet sense counts and lexicographer file, and a PCA of BERT's
static input embedding. The term threshold is 0.5 (balanced class weights),
fixed before any evaluation.
"""

import csv
import random
from functools import lru_cache

import numpy as np
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

from ..data import RAW as V7_RAW
from ..features import base_features

NOUN_LEX = sorted({s.lexname() for s in wn.all_synsets("n")})


@lru_cache(100000)
def type_features(word):
    w = word.lower()
    b = base_features(w)
    f = {"zipf": b["zipf"], "len": b["length"], "syll": b["syllables"]}
    for p in "nvar":
        ss = wn.synsets(w, p)
        f[f"senses_{p}"] = len(ss)
        f[f"count_{p}"] = float(np.log1p(sum(l.count() for s in ss for l in s.lemmas() if l.name().lower() == w)))
    ns = wn.synsets(w, "n")
    first = ns[0].lexname() if ns else "none"
    for ln in NOUN_LEX:
        f[f"lex_{ln}"] = float(first == ln)
    f["topical"] = float(any(s.topic_domains() for s in wn.synsets(w)))
    f["suffix_tech"] = float(w.endswith(("ase", "ine", "ide", "ol", "ate", "ism", "itis", "ology", "ometry", "ics")))
    from .features import bert

    tok, _ = bert()
    f["pieces"] = len(tok(w, add_special_tokens=False).input_ids)
    return f


def static_embedding(words):
    import torch

    from .features import bert

    tok, mlm = bert()
    table = mlm.bert.embeddings.word_embeddings.weight
    out = []
    with torch.no_grad():
        for w in words:
            ids = tok(w.lower(), add_special_tokens=False).input_ids or [tok.unk_token_id]
            out.append(table[ids].mean(0).numpy())
    return np.array(out)


def _matrix(dicts, keys):
    return np.array([[d[k] for k in keys] for d in dicts], dtype=float)


class TermWord:
    def __init__(self, components=16, seed=4701):
        self.components, self.seed = components, seed

    def features(self, words):
        base = [type_features(w) for w in words]
        emb = self.pca.transform(static_embedding(words))
        return np.hstack([_matrix(base, self.keys), emb])

    def fit(self, words, labels):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.keys = sorted(type_features("example"))
        self.pca = PCA(self.components, random_state=self.seed).fit(static_embedding(words))
        X = self.features(words)
        y = np.asarray(labels)
        w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                    l2_regularization=1.0, random_state=self.seed)
        self.model.fit(X, y, sample_weight=w)
        return self

    def predict(self, words):
        return self.model.predict_proba(self.features(list(words)))[:, 1]


def term_training_set(glossaries, replaceable, mesh_cap=1500, seed=4701):
    """(words, labels, source) for TermWord. `replaceable`: words with attested
    simpler substitutes (label 0)."""
    neg = {w.lower() for w in replaceable if w.isalpha()}
    rng = random.Random(seed)
    words, labels, source = [], [], []
    for name in sorted(glossaries):
        terms = sorted(t for t in glossaries[name] if t not in neg)
        if name == "mesh" and len(terms) > mesh_cap:
            terms = sorted(rng.sample(terms, mesh_cap))
        for t in terms:
            words.append(t), labels.append(1), source.append(name)
    for w in sorted(neg):
        words.append(w), labels.append(0), source.append("replaceable")
    seen, out = set(), ([], [], [])
    for w, l, s in zip(words, labels, source):
        if w in seen:
            continue
        seen.add(w)
        out[0].append(w), out[1].append(l), out[2].append(s)
    return out


# --- complex words (CWI 2018) ---------------------------------------------------

def cwi_rows(split):
    rows = []
    with open(V7_RAW / f"cwi_english_{split}.tsv", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) < 10 or " " in r[4] or not r[4].isalpha():
                continue
            rows.append({"text": r[1], "start": int(r[2]), "end": int(r[3]), "word": r[4], "label": int(r[9])})
    return rows


def context_features(doc, token):
    f = dict(type_features(token.text))
    f["lemma_zipf"] = zipf_frequency(token.lemma_.lower(), "en")
    near = [t for t in doc[max(0, token.i - 3):token.i + 4] if t.is_alpha and t.i != token.i]
    f["local_zipf"] = float(np.mean([zipf_frequency(t.text.lower(), "en") for t in near])) if near else 0.0
    for p in ("NOUN", "VERB", "ADJ", "ADV", "PROPN"):
        f[f"upos_{p}"] = float(token.pos_ == p)
    f["ent"] = float(bool(token.ent_type_))
    f["cap"] = float(token.text[:1].isupper() and token.i > 0)
    return f


class ComplexWord:
    def fit(self, dicts, labels, seed=4701):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.keys = sorted(dicts[0])
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                    l2_regularization=1.0, random_state=seed)
        self.model.fit(_matrix(dicts, self.keys), np.asarray(labels))
        self.threshold = 0.5
        return self

    def predict(self, dicts):
        return self.model.predict_proba(_matrix(dicts, self.keys))[:, 1]
