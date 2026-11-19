"""Stage 4b: a learned per-edit checker for sentence splits.

One split edit = one source sentence and the sentences that replace it.
Features: small-NLI entailment source -> output, output -> source and source
-> each new sentence (minimum); parse features of every new sentence
(finite verb, subject, first-word category, closing punctuation, length);
content-word coverage of the source and content words the split adds.

Training data: WikiSplit tune pairs (human splits mined from Wikipedia
edits, kept only when both sentences start with a capital and have a finite
verb, since the corpus contains fragments) as positives, and synthetic corruptions of those splits as
negatives, each labelled with its corruption type:
  bad_boundary   cut the source at a random non-clause point
  dropped_span   delete 3-8 words from the second sentence
  lost_subject   delete the subject of the second sentence
  wrong_referent change the second sentence's leading pronoun
Synthetic negatives are a training device only; the real test of this
checker is human marks on F2's own splits.
"""

import random
import re

import numpy as np

from ..features import nlp
from .features import entail

PRON_SWAP = {"he": "she", "she": "he", "it": "they", "they": "it", "his": "her", "her": "his",
             "its": "their", "their": "its", "this": "these", "these": "this"}
FIRST = ("PRON", "DET", "NOUN", "PROPN", "VERB", "AUX", "ADP", "SCONJ", "CCONJ", "ADV", "ADJ", "NUM", "PART")
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}


def detok(text):
    t = text.replace(" <::::> ", " ")
    t = re.sub(r" ([,.;:!?%)])", r"\1", t)
    t = re.sub(r"([(]) ", r"\1", t)
    t = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", t)
    t = t.replace("`` ", '"').replace(" ''", '"').replace("''", '"')
    return re.sub(r"\s+", " ", t).strip()


def wikisplit(path, limit=None, seed=4701):
    rows = []
    for line in open(path, encoding="utf8"):
        p = line.rstrip("\n").split("\t")
        if len(p) != 2 or "<::::>" not in p[1]:
            continue
        parts = [detok(x) for x in p[1].split("<::::>")]
        if len(parts) != 2 or min(len(x.split()) for x in parts) < 4:
            continue
        rows.append({"source": detok(p[0]), "parts": parts})
    rng = random.Random(seed)
    rng.shuffle(rows)
    # WikiSplit is mined from edit history and contains fragments ("Including
    # a Henderson powered high-wing."). Keep human splits whose two sentences
    # each start with a capital and contain a finite verb.
    kept = []
    for i in range(0, len(rows), 256):
        chunk = rows[i:i + 256]
        docs = list(nlp().pipe([x for r in chunk for x in r["parts"]], batch_size=128))
        for j, r in enumerate(chunk):
            a, b = docs[2 * j], docs[2 * j + 1]
            if all(d.text[:1].isupper() and any(t.tag_ in ("VBD", "VBZ", "VBP", "MD") for t in d) for d in (a, b)):
                kept.append(r)
        if limit and len(kept) >= limit:
            break
    return kept[:limit] if limit else kept


def _cap(s):
    return s[:1].upper() + s[1:]


def corrupt(row, kind, rng):
    src, (a, b) = row["source"], row["parts"]
    if kind == "bad_boundary":
        words = src.rstrip(".").split()
        if len(words) < 10:
            return None
        cut = rng.randint(int(len(words) * 0.3), int(len(words) * 0.7))
        return [" ".join(words[:cut]).rstrip(",;:") + ".", _cap(" ".join(words[cut:]).lstrip(",;: ")) + "."]
    if kind == "dropped_span":
        w = b.split()
        if len(w) < 8:
            return None
        n = rng.randint(3, min(8, len(w) - 3))
        i = rng.randint(1, len(w) - n - 1)
        return [a, " ".join(w[:i] + w[i + n:])]
    doc = nlp()(b)
    if kind == "lost_subject":
        subj = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass") and t.head.dep_ == "ROOT"]
        if not subj:
            return None
        span = doc[subj[0].left_edge.i:subj[0].right_edge.i + 1]
        out = (b[:span.start_char] + b[span.end_char:]).strip()
        return [a, _cap(re.sub(r"\s+", " ", out))] if len(out.split()) >= 3 else None
    if kind == "wrong_referent":
        first = doc[0].text.lower()
        if first not in PRON_SWAP:
            return None
        return [a, _cap(PRON_SWAP[first]) + b[len(doc[0].text):]]
    raise ValueError(kind)


KINDS = ("bad_boundary", "dropped_span", "lost_subject", "wrong_referent")


def build_pairs(rows, seed=4701, per_source=2):
    rng = random.Random(seed)
    out = []
    for i, r in enumerate(rows):
        out.append({"group": i, "source": r["source"], "parts": r["parts"], "label": 1, "kind": "human"})
        kinds = list(KINDS)
        rng.shuffle(kinds)
        made = 0
        for k in kinds:
            if made >= per_source:
                break
            c = corrupt(r, k, rng)
            if c:
                out.append({"group": i, "source": r["source"], "parts": c, "label": 0, "kind": k})
                made += 1
    return out


def sentence_features(doc):
    toks = [t for t in doc if not t.is_space]
    finite = any(t.tag_ in ("VBD", "VBZ", "VBP", "MD") for t in toks)
    root = next((t for t in toks if t.dep_ == "ROOT"), None)
    subj = any(t.dep_ in ("nsubj", "nsubjpass", "expl", "csubj") and t.head == root for t in toks)
    f = {"finite": float(finite), "subject": float(subj), "root_verb": float(bool(root) and root.pos_ in ("VERB", "AUX")),
         "end_punct": float(bool(toks) and toks[-1].text in ".!?\"'"), "n": len(toks),
         "lower_start": float(bool(toks) and toks[0].text[:1].islower()),
         "vbg_root": float(bool(root) and root.tag_ in ("VBG", "VBN") and not any(c.dep_ in ("aux", "auxpass") for c in root.children))}
    for p in FIRST:
        f[f"first_{p}"] = float(bool(toks) and toks[0].pos_ == p)
    f["first_wh"] = float(bool(toks) and toks[0].tag_ in ("WDT", "WP", "WRB"))
    return f


def features(pairs, batch=32):
    """Feature dicts for [{"source", "parts"}]."""
    docs = list(nlp().pipe([s for p in pairs for s in [p["source"]] + p["parts"]], batch_size=64))
    nli_in = []
    for p in pairs:
        out = " ".join(p["parts"])
        nli_in += [(p["source"], out), (out, p["source"])] + [(p["source"], s) for s in p["parts"]]
    ent = entail(nli_in, batch=batch)
    rows, k, d = [], 0, 0
    for p in pairs:
        m = len(p["parts"])
        src, parts = docs[d], docs[d + 1:d + 1 + m]
        d += 1 + m
        e = ent[k:k + 2 + m]
        k += 2 + m
        sf = [sentence_features(x) for x in parts]
        src_c = {t.lemma_.lower() for t in src if t.pos_ in CONTENT}
        out_c = {t.lemma_.lower() for x in parts for t in x if t.pos_ in CONTENT}
        f = {"nli_fw": e[0], "nli_bw": e[1], "nli_part_min": min(e[2:]), "nli_part_mean": float(np.mean(e[2:])),
             "parts": m, "coverage": len(src_c & out_c) / max(1, len(src_c)), "added": len(out_c - src_c),
             "len_ratio": sum(len(x) for x in parts) / max(1, len(src)), "src_len": len(src),
             "min_part_len": min(len(x) for x in parts)}
        for key in sf[0]:
            vals = [s[key] for s in sf]
            f[f"min_{key}"] = float(min(vals))
            f[f"max_{key}"] = float(max(vals))
        rows.append(f)
    return rows


class SplitChecker:
    def fit(self, dicts, labels, seed=4701):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.keys = sorted(dicts[0])
        X = np.array([[d[k] for k in self.keys] for d in dicts])
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                    l2_regularization=1.0, random_state=seed)
        self.model.fit(X, np.asarray(labels))
        return self

    def predict(self, dicts):
        if not dicts:
            return np.array([])
        return self.model.predict_proba(np.array([[d[k] for k in self.keys] for d in dicts]))[:, 1]
