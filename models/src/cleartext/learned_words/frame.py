"""Collocation / verb-frame damage of a one-word swap, from BERT's masked LM.

Round-4 meaning errors were mostly broken frames and collocations ("we will
be needed to", "asking an extension", "clinically important"). The checker's
features look at the swapped slot; these look at its neighbours: mask each
word within ``RADIUS`` words of the slot, before and after the swap, and
measure how much less likely BERT finds the neighbour once the replacement
is in place. Also the slot fit itself (the replacement's masked
log-probability, first piece, minus the target's).
"""

import re

import numpy as np

RADIUS = 3


def _words(text):
    return [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z][A-Za-z'-]*|\d+|[^\sA-Za-z\d]", text)]


def _masked_logp(texts_spans):
    """log P(original text at span | text with that span masked), first wordpiece."""
    import torch

    from .features import bert

    tok, mlm = bert()
    out = []
    for i in range(0, len(texts_spans), 16):
        batch = texts_spans[i:i + 16]
        masked, ids = [], []
        for text, (a, b) in batch:
            masked.append(text[:a] + tok.mask_token + text[b:])
            piece = tok(text[a:b].lower(), add_special_tokens=False).input_ids
            ids.append(piece[0] if piece else tok.unk_token_id)
        enc = tok(masked, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            logits = mlm(**enc).logits
        lp = torch.log_softmax(logits, -1)
        for j, wid in enumerate(ids):
            pos = (enc.input_ids[j] == tok.mask_token_id).nonzero()
            out.append(float(lp[j, pos[0, 0], wid]) if len(pos) else float("nan"))
    return out


def frame_features(before, after, start, end_before, end_after):
    """Features for one swap; [start, end_*) is the target / replacement span."""
    wb, wa = _words(before), _words(after)
    ib = next(i for i, (a, _) in enumerate(wb) if a == start)
    ia = next(i for i, (a, _) in enumerate(wa) if a == start)
    pairs = []
    for k in range(-RADIUS, RADIUS + 1):
        if k == 0 or not (0 <= ib + k < len(wb)) or not (0 <= ia + k < len(wa)):
            continue
        sb, sa = wb[ib + k], wa[ia + k]
        if not before[sb[0]:sb[1]][:1].isalpha():
            continue
        pairs.append((k, (before, sb), (after, sa)))
    lps = _masked_logp([p[1] for p in pairs] + [p[2] for p in pairs]
                       + [(before, (start, end_before)), (after, (start, end_after))])
    n = len(pairs)
    drops = np.array([lps[i] - lps[n + i] for i in range(n)]) if n else np.zeros(1)
    near = np.array([lps[i] - lps[n + i] for i, p in enumerate(pairs) if abs(p[0]) == 1]) if n else np.zeros(1)
    return {"nb_max_drop": float(drops.max()), "nb_sum_drop": float(np.clip(drops, 0, None).sum()),
            "nb_mean_drop": float(drops.mean()), "nb_adj_max_drop": float(near.max()) if len(near) else 0.0,
            "slot_d_logp": float(lps[-1] - lps[-2]), "slot_rep_logp": float(lps[-1])}


def swap_span(before, after, word, rep):
    """(start, end_before, end_after) of the swapped word."""
    i = 0
    while i < min(len(before), len(after)) and before[i] == after[i]:
        i += 1
    start = before.rfind(word, 0, i + len(word))
    if start < 0 or after[start:start + len(rep)] != rep:
        m = re.search(r"\b%s\b" % re.escape(word), before)
        start = m.start()
    return start, start + len(word), start + len(rep)
