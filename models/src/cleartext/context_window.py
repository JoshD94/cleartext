"""Opt-in BERT meaning vectors from a window that contains the complete target."""

from dataclasses import dataclass
import operator

import numpy as np

from .ensemble import BertSense
from .lexical import synsets
from .wsd import word_pos


@dataclass(frozen=True)
class TargetWindow:
    input_ids: tuple
    target_positions: tuple
    source_start: int
    source_end: int
    source_length: int


def target_windows(tokenizer, words, indices, limit=256):
    """Return one bounded window per requested word, preserving request order.

    Bounds refer to the original wordpiece sequence without special tokens.
    Target positions refer to the returned input, including its CLS token.
    This helper is specific to the existing BERT CLS/SEP input format.
    """
    if limit < 3:
        raise ValueError("window limit must leave space for CLS, target and SEP")
    indices = [operator.index(index) for index in indices]
    if any(index < 0 or index >= len(words) for index in indices):
        raise IndexError("target word index is outside the input")
    if not indices:
        return []
    encoded = tokenizer(
        list(words),
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )
    ids = encoded.input_ids
    positions = {index: [] for index in indices}
    for piece, word in enumerate(encoded.word_ids()):
        if word in positions:
            positions[word].append(piece)
    budget = limit - 2
    windows = []
    for index in indices:
        target = positions[index]
        if not target:
            raise ValueError(f"target word at index {index} has no wordpieces")
        start, end = target[0], target[-1] + 1
        if end - start > budget:
            raise ValueError(f"complete target at index {index} exceeds window budget")
        left = max(0, start - (budget - (end - start)) // 2)
        right = min(len(ids), left + budget)
        left = max(0, right - budget)
        windows.append(
            TargetWindow(
                tuple(
                    [tokenizer.cls_token_id]
                    + ids[left:right]
                    + [tokenizer.sep_token_id]
                ),
                tuple(piece - left + 1 for piece in target),
                left,
                right,
                len(ids),
            )
        )
    return windows


def windowed_word_vectors(sentences, limit=256):
    """Same output shape and pooling as contextual.word_vectors, without lost targets."""
    import torch
    from .contextual import model

    tokenizer, mlm = model()
    if limit > mlm.config.max_position_embeddings:
        raise ValueError("window exceeds the encoder position limit")
    output, pending = [], []
    for words, indices in sentences:
        windows = target_windows(tokenizer, words, indices, limit)
        owner = len(output)
        output.append([None] * len(windows))
        pending.extend((owner, index, window) for index, window in enumerate(windows))
    with torch.no_grad():
        for start in range(0, len(pending), 16):
            chunk = pending[start : start + 16]
            length = max(len(window.input_ids) for _, _, window in chunk)
            ids = torch.tensor(
                [
                    list(window.input_ids)
                    + [tokenizer.pad_token_id] * (length - len(window.input_ids))
                    for _, _, window in chunk
                ]
            )
            attention = torch.tensor(
                [
                    [1] * len(window.input_ids) + [0] * (length - len(window.input_ids))
                    for _, _, window in chunk
                ]
            )
            hidden = mlm.bert(
                input_ids=ids, attention_mask=attention, output_hidden_states=True
            ).hidden_states
            pooled = torch.stack(hidden[-4:]).mean(0)
            for row, (owner, index, window) in enumerate(chunk):
                vector = pooled[row, list(window.target_positions)].mean(0).numpy()
                output[owner][index] = vector / (np.linalg.norm(vector) or 1)
    return output


def windowed_word_vector(words, index):
    return windowed_word_vectors([(list(words), [index])])[0][0]


class WindowedBertSense(BertSense):
    """Reuse frozen sense vectors and temperature, with a visible contextual target."""

    name = "bert_sense_windowed"

    def similarities(self, words, index, lemma, pos):
        senses = synsets(lemma, word_pos(pos))
        if not senses:
            return {}
        vector = windowed_word_vector(words, index)
        return {
            sense.name(): float(vector @ self.sense_vector(sense.name()))
            for sense in senses
        }
