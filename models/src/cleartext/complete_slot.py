"""Opt-in BERT slot scores that retain every candidate wordpiece."""

from dataclasses import dataclass

import numpy as np

from . import contextual, ensemble as E


@dataclass(frozen=True)
class SlotInput:
    input_ids: tuple
    positions: tuple
    pieces: tuple


def slot_inputs(tokenizer, context, start, end, words, window=150, limit=320):
    if not 0 <= start < end <= len(context):
        raise ValueError(
            "target offsets must identify a nonempty span within the context"
        )
    if window < 0 or limit < 3:
        raise ValueError("invalid context window or token limit")
    before = tokenizer(context[:start], add_special_tokens=False).input_ids
    after = tokenizer(context[end:], add_special_tokens=False).input_ids
    result = []
    for word in words:
        pieces = tokenizer(
            (" " if start else "") + word, add_special_tokens=False
        ).input_ids
        if not pieces:
            raise ValueError("candidate has no wordpieces")
        budget = limit - 2 - len(pieces)
        if budget < 0:
            raise ValueError("complete candidate cannot fit within the token limit")
        left, right = len(before), len(after)
        if left + right > budget:
            left = min(len(before), window, budget // 2)
            right = min(len(after), window, budget - left)
            left = min(len(before), window, budget - right)
        ids = (
            [tokenizer.cls_token_id]
            + (before[-left:] if left else [])
            + pieces
            + after[:right]
            + [tokenizer.sep_token_id]
        )
        result.append(
            SlotInput(
                tuple(ids),
                tuple(range(1 + left, 1 + left + len(pieces))),
                tuple(pieces),
            )
        )
    return result


def complete_slot_scores(
    context, start, end, words, window=150, limit=320, batch_size=16
):
    """Mean masked log-probability over all pieces, with bounded encoder batches.

    Apply the vocabulary head only at masked positions. This avoids allocating
    a full sequence-by-vocabulary tensor for each masked candidate piece.
    """
    import torch

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if not words:
        return np.array([])
    tokenizer, model = contextual.model()
    limit = min(limit, model.config.max_position_embeddings)
    inputs = slot_inputs(tokenizer, context, start, end, words, window, limit)
    totals = np.zeros(len(words))
    counts = np.zeros(len(words))

    def jobs():
        for owner, item in enumerate(inputs):
            for position, piece in zip(item.positions, item.pieces):
                ids = list(item.input_ids)
                ids[position] = tokenizer.mask_token_id
                yield owner, ids, position, piece

    def score(chunk):
        length = max(len(ids) for _, ids, _, _ in chunk)
        ids = torch.tensor(
            [
                ids + [tokenizer.pad_token_id] * (length - len(ids))
                for _, ids, _, _ in chunk
            ]
        )
        mask = torch.tensor(
            [[1] * len(ids) + [0] * (length - len(ids)) for _, ids, _, _ in chunk]
        )
        hidden = model.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        rows = torch.arange(len(chunk))
        positions = torch.tensor([position for _, _, position, _ in chunk])
        probabilities = model.cls(hidden[rows, positions]).log_softmax(-1)
        for row, (owner, _, _, piece) in enumerate(chunk):
            totals[owner] += float(probabilities[row, piece])
            counts[owner] += 1

    with torch.no_grad():
        chunk = []
        for job in jobs():
            chunk.append(job)
            if len(chunk) == batch_size:
                score(chunk)
                chunk = []
        if chunk:
            score(chunk)
    return totals / counts


def cached_scores(slot, candidates):
    key = ("complete_bert_slot", tuple(candidate["word"] for candidate in candidates))
    if key not in slot.cache:
        token = slot.token
        scores = complete_slot_scores(
            slot.doc.text,
            token.idx,
            token.idx + len(token),
            [candidate["word"] for candidate in candidates] + [token.text],
        )
        slot.cache[key] = scores[:-1], scores[-1]
    return slot.cache[key]


class CompleteBertSlotFit(E.BertSlotFit):
    def score(self, slot, candidates):
        return cached_scores(slot, candidates)[0] if candidates else np.array([])


class CompleteBertSlotRelative(E.BertSlotRelative):
    def score(self, slot, candidates):
        if not candidates:
            return np.array([])
        scores, original = cached_scores(slot, candidates)
        return scores - original
