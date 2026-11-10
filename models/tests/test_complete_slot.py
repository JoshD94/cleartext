import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext import contextual, ensemble as E
from cleartext.complete_slot import (
    slot_inputs,
    complete_slot_scores,
    CompleteBertSlotFit,
)
from cleartext.ensemble_pipeline import fit_members


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)


def test_entire_candidate_survives_long_context_and_exact_offset(tokenizer):
    context = "record " * 400 + "target" + " record" * 400
    start = context.index("target")
    phrase = "one two three four five six seven eight nine"
    item = slot_inputs(tokenizer, context, start, start + 6, [phrase])[0]
    expected = tokenizer(phrase, add_special_tokens=False).input_ids
    assert len(expected) > 6
    assert list(item.pieces) == expected
    assert [item.input_ids[position] for position in item.positions] == expected
    assert len(item.input_ids) <= 320
    tight = slot_inputs(tokenizer, context, start, start + 6, [phrase], limit=12)[0]
    assert len(tight.input_ids) <= 12 and tight.pieces == item.pieces


def test_invalid_candidates_and_offsets_fail_explicitly(tokenizer):
    with pytest.raises(ValueError, match="complete candidate"):
        slot_inputs(tokenizer, "target", 0, 6, ["one two three"], limit=4)
    with pytest.raises(ValueError, match="no wordpieces"):
        slot_inputs(tokenizer, "target", 0, 6, [""])
    with pytest.raises(ValueError, match="offsets"):
        slot_inputs(tokenizer, "target", -1, 6, ["word"])


def test_every_piece_is_scored_in_bounded_batches_and_literal_pad_is_visible(
    tokenizer, monkeypatch
):
    import torch

    batches = []

    def encoder(input_ids, attention_mask):
        batches.append((input_ids.clone(), attention_mask.clone()))
        return SimpleNamespace(last_hidden_state=torch.ones((*input_ids.shape, 2)))

    def head(hidden):
        assert hidden.ndim == 2 and hidden.shape[0] <= 2
        return torch.zeros((len(hidden), tokenizer.vocab_size))

    model = SimpleNamespace(
        bert=encoder, cls=head, config=SimpleNamespace(max_position_embeddings=512)
    )
    monkeypatch.setattr(contextual, "model", lambda: (tokenizer, model))
    words = ["one two three four five six seven", "small"]
    context = "[PAD] target"
    result = complete_slot_scores(context, 6, 12, words, batch_size=2)
    assert sum(len(ids) for ids, _ in batches) == 8
    assert len(batches) == 4
    for ids, mask in batches:
        assert torch.all(mask[:, 1] == 1)
        assert torch.all(ids[:, 1] == tokenizer.pad_token_id)
    np.testing.assert_allclose(result, -np.log(tokenizer.vocab_size), atol=1e-6)
    assert complete_slot_scores("", 0, 0, []).size == 0


def test_complete_members_require_opt_in_and_keep_frozen_feature_names():
    old = fit_members(None, None, bert=True)
    new = fit_members(None, None, bert=True, complete_slot=True)
    assert type(old[-2]) is E.BertSlotFit
    assert type(new[-2]) is CompleteBertSlotFit
    assert [member.name for member in old] == [member.name for member in new]
