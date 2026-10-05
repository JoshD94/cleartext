"""Contextual similarity of the original span and a complete replacement span.

This is an encoder signal, not a meaning-preservation guarantee. In particular,
antonyms and broader terms can have similar contextual representations.
"""

from functools import lru_cache

import numpy as np

from . import contextual
from .complete_slot import slot_inputs
from .prompted_relations import features_prompted_relations


def slot_vectors(inputs, tokenizer, model, batch_size=16):
    """Unit vectors from the last four layers, pooled over every span piece."""
    import torch

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    result = []
    with torch.no_grad():
        for offset in range(0, len(inputs), batch_size):
            chunk = inputs[offset : offset + batch_size]
            length = max(len(item.input_ids) for item in chunk)
            ids = torch.tensor(
                [
                    list(item.input_ids)
                    + [tokenizer.pad_token_id] * (length - len(item.input_ids))
                    for item in chunk
                ]
            )
            mask = torch.tensor(
                [
                    [1] * len(item.input_ids) + [0] * (length - len(item.input_ids))
                    for item in chunk
                ]
            )
            states = model.bert(
                input_ids=ids, attention_mask=mask, output_hidden_states=True
            ).hidden_states
            hidden = torch.stack(states[-4:]).mean(0)
            for row, item in enumerate(chunk):
                vector = hidden[row, list(item.positions)].mean(0).numpy()
                norm = np.linalg.norm(vector)
                if not np.isfinite(norm) or norm == 0:
                    raise ValueError("encoder produced an invalid span vector")
                result.append(vector / norm)
    return np.asarray(result)


def replacement_similarities(context, start, end, words, batch_size=16, limit=256):
    if not words:
        return np.array([])
    tokenizer, model = contextual.model()
    inputs = slot_inputs(
        tokenizer,
        context,
        start,
        end,
        [context[start:end]] + list(words),
        limit=min(limit, model.config.max_position_embeddings),
    )
    vectors = slot_vectors(inputs, tokenizer, model, batch_size)
    return np.clip(vectors[1:] @ vectors[0], -1.0, 1.0)


@lru_cache(1024)
def similarity_for(context, start, end, words):
    return tuple(replacement_similarities(context, start, end, words))


def features_prompted_similarity(case):
    if not case["candidates"]:
        return lambda case, candidate: None
    base = features_prompted_relations(case)
    scores = case.get("replacement_similarity")
    if scores is None:
        start = case.get("start")
        if (
            start is None
            or case["text"][start : start + len(case["target"])].casefold()
            != case["target"].casefold()
        ):
            raise ValueError("replacement similarity requires the exact target span")
        words = tuple(candidate["word"] for candidate in case["candidates"])
        scores = dict(
            zip(
                words,
                similarity_for(case["text"], start, start + len(case["target"]), words),
            )
        )

    def row(case, candidate):
        return np.r_[base(case, candidate), scores[candidate["word"]]]

    return row
