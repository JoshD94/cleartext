"""Sense match and parse stability for one word swap (round 9).

A swap keeps its meaning when the replacement, read in the new sentence,
expresses the same WordNet sense the target expressed in the old one.

- Sense vectors: BERT-base (the stage's own masked LM), sum of the last four
  layers, averaged over each SemCor token tagged with that synset (LMMS-style),
  blended with a vector of the lemma in its WordNet examples or gloss, so every
  synset has one. Built once by `build` (checkpointed).
- WSD: softmax over the lemma's synsets of cosine(context vector, sense vector)
  at a fixed temperature.
- Parse stability: re-parse the swapped sentence and compare the replaced
  token's part of speech, dependency, head and children with the target's.
"""

import pickle
import time
from functools import lru_cache

import numpy as np

TAU = 0.05
LAYERS = 4
BLEND_PRIOR = 2.0  # semcor weight n / (n + 2)
CAP = 20  # SemCor tokens per synset; a sentence whose synsets are all full is skipped
WN_POS = {"n": "n", "v": "v", "a": "a", "r": "r"}
SENSE_KEYS = ("sense_same", "sense_t_cov", "sense_r_cov", "sense_top", "cross_fit", "rep_has_syn", "sense_hyper")
PARSE_KEYS = ("pos_change", "dep_change", "head_change", "kids_change", "tree_diff", "multiword")
KEYS = SENSE_KEYS + PARSE_KEYS


def _bert():
    from . import features as F

    return F.bert()


def contextual_multi(texts, spans, batch=16):
    """texts[i] with character spans[i] -> list of unit-vector arrays, one row per span
    (sum of last 4 layers, mean over the span's word pieces). One forward pass per text."""
    import torch

    tok, mlm = _bert()
    out = []
    for i in range(0, len(texts), batch):
        chunk, cs = texts[i:i + batch], spans[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128,
                  return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        with torch.no_grad():
            hs = mlm.bert(**enc, output_hidden_states=True).hidden_states
        h = torch.stack(hs[-LAYERS:]).sum(0)
        for j, sp in enumerate(cs):
            o = offs[j].tolist()
            rows = []
            for a, b in sp:
                sel = [k for k, (x, y) in enumerate(o) if y > x and x < b and y > a]
                v = h[j, sel].mean(0).numpy() if sel else np.full(h.shape[-1], np.nan, np.float32)
                n = np.linalg.norm(v)
                rows.append(v / n if n > 0 else v)
            out.append(np.array(rows, np.float32).reshape(len(sp), h.shape[-1]))
    return out


def contextual(items, batch=16):
    """items: (text, start, end) character spans -> unit vectors, one per item."""
    return np.concatenate([m for m in contextual_multi([t for t, _, _ in items], [[(a, b)] for _, a, b in items],
                                                       batch=batch)])


# --- sense table -------------------------------------------------------------------

def semcor_items():
    """(text, [(start, end, synset_name)]) per SemCor sentence."""
    from nltk.corpus import semcor
    from nltk.tree import Tree

    for sent in semcor.tagged_sents(tag="sem"):
        words, spans, pos = [], [], 0
        for chunk in sent:
            if isinstance(chunk, Tree):
                text = " ".join(chunk.leaves())
                label = chunk.label()
                syn = None
                if hasattr(label, "synset"):
                    try:
                        syn = label.synset().name()
                    except Exception:
                        syn = None
            else:
                text = " ".join(chunk) if isinstance(chunk, list) else str(chunk)
                syn = None
            if pos:
                pos += 1
            words.append(text)
            if syn:
                spans.append((pos, pos + len(text), syn))
            pos += len(text)
        yield " ".join(words), spans


def build(path, max_seconds=1500, batch=16):
    """Accumulate SemCor sense sums into `path` (pickle); resumable. Returns True when complete."""
    state = pickle.load(open(path, "rb")) if path.exists() else {"next": 0, "sum": {}, "n": {}, "done": False}
    if state["done"]:
        return True
    deadline = time.monotonic() + max_seconds
    pending = []

    def flush():
        vecs = contextual_multi([t for t, _ in pending], [[(a, b) for a, b, _ in sp] for _, sp in pending], batch=batch)
        for (_, sp), V in zip(pending, vecs):
            for (_, _, syn), v in zip(sp, V):
                if np.isfinite(v).all():
                    state["sum"][syn] = state["sum"].get(syn, 0) + v.astype(np.float64)
                    state["n"][syn] = state["n"].get(syn, 0) + 1
        pending.clear()

    k = -1
    for k, (text, spans) in enumerate(semcor_items()):
        if k < state["next"]:
            continue
        seen = state.setdefault("queued", {})
        if any(seen.get(syn, 0) < CAP for _, _, syn in spans):
            pending.append((text, spans))
            for _, _, syn in spans:
                seen[syn] = seen.get(syn, 0) + 1
        if (k + 1) % 200 == 0:
            flush()
            state["next"] = k + 1
            pickle.dump(state, open(path, "wb"))
            if time.monotonic() > deadline:
                print(f"PARTIAL at sentence {k + 1}, {len(state['n'])} synsets", flush=True)
                return False
    flush()
    state.update(next=k + 1, done=True)
    pickle.dump(state, open(path, "wb"))
    print(f"COMPLETE {k + 1} sentences, {len(state['n'])} synsets", flush=True)
    return True


class Senses:
    def __init__(self, table_path):
        state = pickle.load(open(table_path, "rb"))
        if not state["done"]:
            raise SystemExit("REFUSED: sense table incomplete")
        self.semcor = {s: (v / np.linalg.norm(v)).astype(np.float32) for s, v in state["sum"].items()}
        self.count = state["n"]
        self._gloss = {}

    def gloss_vec(self, syn, lemma):
        key = (syn.name(), lemma)
        if key not in self._gloss:
            word = lemma.replace("_", " ")
            ex = [e for e in syn.examples() if word.lower() in e.lower()]
            if ex:
                t = ex[0]
                a = t.lower().find(word.lower())
            else:
                t = f"{word} : {syn.definition()}"
                a = 0
            self._gloss[key] = contextual([(t, a, a + len(word))])[0]
        return self._gloss[key]

    def vec(self, syn, lemma):
        g = self.gloss_vec(syn, lemma)
        s = self.semcor.get(syn.name())
        if s is None:
            return g
        w = self.count[syn.name()] / (self.count[syn.name()] + BLEND_PRIOR)
        v = w * s + (1 - w) * np.nan_to_num(g)
        return v / np.linalg.norm(v)

    def wsd(self, ctx, lemma, pos):
        """{synset: prob} for `lemma` read with unit context vector `ctx`."""
        from nltk.corpus import wordnet as wn

        allowed = ("a", "s") if pos == "a" else (WN_POS[pos],)
        syns = [s for s in wn.synsets(lemma) if s.pos() in allowed]
        if not syns or not np.isfinite(ctx).all():
            return {}, {}
        V = {s: self.vec(s, lemma) for s in syns}
        c = np.array([float(ctx @ V[s]) for s in syns])
        p = np.exp((c - c.max()) / TAU)
        p /= p.sum()
        return dict(zip(syns, p)), V


def _lemma(tokens, fallback):
    if len(tokens) == 1:
        return tokens[0].lemma_.lower()
    return fallback.lower().replace(" ", "_")


def swap_features(senses, before, after, start, word, rep, pos):
    """Sense-match and parse-stability features of one resolved swap."""
    from cleartext.features import nlp

    d0, d1 = nlp()(before), nlp()(after)
    t = next(x for x in d0 if x.idx == start)
    rt = [x for x in d1 if start <= x.idx < start + len(rep)] or [d1[min(t.i, len(d1) - 1)]]
    r0 = rt[0]
    c_t, c_r = contextual([(before, start, start + len(word)), (after, start, start + len(rep))])
    tl, rl = t.lemma_.lower(), _lemma(rt, rep)
    pt, Vt = senses.wsd(c_t, tl, pos)
    pr, _ = senses.wsd(c_r, rl, pos)
    if not pr and rl != rep.lower().replace(" ", "_"):
        pr, _ = senses.wsd(c_r, rep.lower().replace(" ", "_"), pos)
    shared = set(pt) & set(pr)
    f = {"sense_same": float(sum(pt[s] * pr[s] for s in shared)),
         "sense_t_cov": float(sum(pt[s] for s in shared)),
         "sense_r_cov": float(sum(pr[s] for s in shared)),
         "rep_has_syn": float(bool(pr))}
    if pt:
        top = max(pt, key=pt.get)
        f["sense_top"] = float(top in pr)
        f["cross_fit"] = float(c_r @ Vt[top] - c_t @ Vt[top]) if np.isfinite(c_r).all() else -1.0
        hyp = {h for h in top.closure(lambda z: z.hypernyms() + z.instance_hypernyms(), depth=2)}
        f["sense_hyper"] = float(bool(set(pr) & hyp) and top not in pr)
    else:
        f.update(sense_top=0.0, cross_fit=0.0, sense_hyper=0.0)
    same_len = len(d0) == len(d1)
    f.update({"pos_change": float(r0.pos_ != t.pos_), "dep_change": float(r0.dep_ != t.dep_),
              "head_change": float((t.head.i == t.i) != (r0.head.i == r0.i)
                                   or (t.head.i != t.i and r0.head.lemma_.lower() != t.head.lemma_.lower())),
              "kids_change": float(sorted(c.dep_ for c in t.children) != sorted(c.dep_ for c in r0.children)),
              "tree_diff": float(min(5, sum((a.dep_, a.head.i) != (b.dep_, b.head.i) for a, b in zip(d0, d1)))
                                 if same_len else 5.0),
              "multiword": float(len(rt) > 1)})
    return {k: f[k] for k in KEYS}


@lru_cache(1)
def load(table_path):
    return Senses(table_path)
