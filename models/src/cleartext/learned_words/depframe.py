"""Dependency-anchored frame features of a one-word swap (round 24), from BERT's masked LM.

Round 5's frame probe masked every word within 3 of the slot and took the
largest probability drop (AUC 0.64 for meaning errors); most of those words
have no syntactic tie to the target, so the signal was noisy. Rounds 6 and
15 used corpus window statistics, which tracked the corpus genre rather than
the frame. Here the probe is anchored on the target's dependency neighbours
(head, children, and the objects of its prepositions), and everything is
scored relative to the target:

- ``drop`` of a neighbour: log P(neighbour | sentence with the target) minus
  log P(neighbour | sentence with the replacement), the neighbour masked.
  "orally administered" predicts "orally" far better than "orally given".
- the neighbour the target predicts best (its strongest collocate) and the
  drop there;
- the slot preference in the bare frame: the target's neighbours alone, in
  order, with the slot masked, log P(replacement) minus log P(target).
  This isolates the collocation from the sentence's general fit.

Learned weights only; nothing here vetoes a swap.
"""

import numpy as np

KEYS = ("dep_max_drop", "dep_sum_drop", "dep_top_drop", "dep_top_lp", "dep_head_drop", "frame_slot_d", "dep_n")
PREP = {"prep", "agent", "dative"}


def neighbours(tok):
    """The target's syntactic neighbours: head, children, and objects of child prepositions."""
    out = []
    if tok.head is not tok:
        out.append(tok.head)
    for c in tok.children:
        out.append(c)
        if c.dep_ in PREP:
            out += [g for g in c.children if g.dep_ in ("pobj", "pcomp")]
    seen, keep = set(), []
    for n in out:
        if n.i in seen or n.i == tok.i or not n.text[:1].isalpha():
            continue
        seen.add(n.i)
        keep.append(n)
    return keep


def frame_keys(doc, start):
    """(relation, neighbour lemma) keys of the target at character `start`: its head as
    ``head:<dep>``, each non-punctuation child as ``child:<dep>``, and the object of a child
    preposition as ``pobj:<preposition>``. These key the learned frame prior (round 24)."""
    tgt = next((t for t in doc if t.idx <= start < t.idx + len(t)), None)
    if tgt is None:
        return []
    out = []
    if tgt.head is not tgt:
        out.append(("head:" + tgt.dep_, tgt.head.lemma_.lower()))
    for c in tgt.children:
        if c.is_punct:
            continue
        out.append(("child:" + c.dep_, c.lemma_.lower()))
        if c.dep_ in PREP:
            out += [("pobj:" + c.lemma_.lower(), g.lemma_.lower()) for g in c.children if g.dep_ in ("pobj", "pcomp")]
    return sorted(set(out))


def _first_piece(tok, word):
    piece = tok(word.lower(), add_special_tokens=False).input_ids
    return piece[0] if piece else tok.unk_token_id


def _masked(texts_spans, words=None, batch=16):
    """For each (text, (a, b)): log-softmax row at the masked span; returns log P of the first piece
    of the original span text, or of each word in `words[i]` when given."""
    import torch

    from .features import bert

    tok, mlm = bert()
    out = []
    for i in range(0, len(texts_spans), batch):
        chunk = texts_spans[i:i + batch]
        masked = [t[:a] + tok.mask_token + t[b:] for t, (a, b) in chunk]
        enc = tok(masked, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            lp = torch.log_softmax(mlm(**enc).logits, -1)
        for j, (t, (a, b)) in enumerate(chunk):
            pos = (enc.input_ids[j] == tok.mask_token_id).nonzero()
            want = words[i + j] if words is not None else [t[a:b]]
            if not len(pos):
                out.append([float("nan")] * len(want))
                continue
            row = lp[j, pos[0, 0]]
            out.append([float(row[_first_piece(tok, w)]) for w in want])
    return out


def _prepare(nlp, before, start, word, rep):
    after = before[:start] + rep + before[start + len(word):]
    doc = nlp(before)
    tgt = next((t for t in doc if t.idx <= start < t.idx + len(t)), None)
    if tgt is None:
        return None
    nb = neighbours(tgt)
    shift = len(rep) - len(word)
    spans = []
    for n in nb:
        a, b = n.idx, n.idx + len(n)
        spans.append(((before, (a, b)), (after, (a + shift, b + shift) if a > start else (a, b))))
    # bare frame: the target and its neighbours in sentence order, the slot masked
    text, slot = "", None
    for t in sorted(nb + [tgt], key=lambda t: t.i):
        if text:
            text += " "
        if t.i == tgt.i:
            slot = (len(text), len(text) + len(t))
        text += t.text
    head = [k for k, x in enumerate(nb) if tgt.head is not tgt and x.i == tgt.head.i]
    return {"spans": spans, "frame": (text, slot), "words": [rep, word], "head": head[0] if head else None}


def _finish(p, got, fr):
    n = len(p["spans"])
    if not n:
        return dict(dict.fromkeys(KEYS, 0.0), frame_slot_d=float(np.nan_to_num(fr[0] - fr[1])))
    lb = np.array([g[0] for g in got[:n]])
    la = np.array([g[0] for g in got[n:]])
    drop = np.nan_to_num(lb - la)
    top = int(np.nanargmax(lb)) if not np.all(np.isnan(lb)) else None
    return {"dep_max_drop": float(drop.max()), "dep_sum_drop": float(np.clip(drop, 0, None).sum()),
            "dep_top_drop": float(drop[top]) if top is not None else 0.0,
            "dep_top_lp": float(lb[top]) if top is not None else 0.0,
            "dep_head_drop": float(drop[p["head"]]) if p["head"] is not None else 0.0,
            "frame_slot_d": float(np.nan_to_num(fr[0] - fr[1])), "dep_n": float(n)}


def features_many(nlp, items):
    """KEYS for each (before, start, word, rep): `word` at character `start` of `before` is replaced
    by `rep`. Masked-LM passes are batched across items."""
    prep = [_prepare(nlp, *it) for it in items]
    ts, frames, fw = [], [], []
    for p in prep:
        if p is not None:
            ts += [s[0] for s in p["spans"]] + [s[1] for s in p["spans"]]
            frames.append(p["frame"])
            fw.append(p["words"])
    got = _masked(ts) if ts else []
    fr = _masked(frames, words=fw) if frames else []
    out, i, j = [], 0, 0
    for p in prep:
        if p is None:
            out.append(dict.fromkeys(KEYS, 0.0))
            continue
        k = 2 * len(p["spans"])
        out.append(_finish(p, got[i:i + k], fr[j]))
        i, j = i + k, j + 1
    return out


def features(nlp, before, start, word, rep):
    return features_many(nlp, [(before, start, word, rep)])[0]
