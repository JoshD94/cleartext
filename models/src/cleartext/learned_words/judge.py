"""A fine-tuned substitution judge: is this one-word swap a good simplification here?

A cross-encoder (``cross-encoder/nli-deberta-v3-small``, NLI-trained, with a
new one-logit head) reads a window of the sentence before the swap and the
same window after it, the swapped word marked in both, and outputs the
probability that the swap keeps the meaning and reads naturally. It is
fine-tuned (``evals/learned_words_judge.py``) on public data only: SWORDS
dev substitutes (soft label = share of annotators who accepted it) and
BenchLS / TSAR-trial candidates (label 1 for a human simplification, 0 for a
generated candidate no annotator gave).

Unlike the stage-4 checker, whose inputs are frozen-model features, the
judge reads the text itself, so it can see a broken verb frame or
collocation ("we will be needed to", "asking an extension").
"""

from functools import lru_cache

import numpy as np

BASE = "cross-encoder/nli-deberta-v3-small"
SIZE = 12
MAX_LEN = 96
OPEN, CLOSE = "[", "]"


def pair(text, start, end, word, size=SIZE):
    """(before window, after window) with the target / replacement marked."""
    left = text[:start].split()[-size:]
    right = text[end:].split()[:size]
    a = " ".join(left + [f"{OPEN}{text[start:end]}{CLOSE}"] + right)
    b = " ".join(left + [f"{OPEN}{word}{CLOSE}"] + right)
    return a, b


def load_base(num_labels=1):
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    transformers.logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=num_labels, ignore_mismatched_sizes=True)
    return tok, model


@lru_cache(2)
def load(path):
    """Tokenizer and fine-tuned judge from a state file (weights only)."""
    import torch

    tok, model = load_base()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return tok, model.eval()


def score(path, pairs, batch=64):
    """P(good swap) for (before window, after window) pairs."""
    import torch

    tok, model = load(str(path))
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            a, b = zip(*pairs[i:i + batch])
            enc = tok(list(a), list(b), return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
            out.append(torch.sigmoid(model(**enc).logits[:, 0]).numpy())
    return np.concatenate(out) if out else np.array([])
