import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.context_window import target_windows, WindowedBertSense
from cleartext import ensemble as E
from cleartext import ensemble_pipeline as pipeline


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)


def test_short_input_keeps_exact_bert_input_and_target_positions(tokenizer):
    words = ["The", "unaffordable", "database", "failed", "."]
    original = tokenizer(words, is_split_into_words=True)
    windows = target_windows(tokenizer, words, [1, 2])
    for target, window in zip([1, 2], windows):
        assert list(window.input_ids) == original.input_ids
        assert list(window.target_positions) == [
            i for i, word in enumerate(original.word_ids()) if word == target
        ]


def test_late_target_and_all_its_wordpieces_survive(tokenizer):
    words = ["record"] * 400 + ["unaffordable", "database"] + ["record"] * 300
    windows = target_windows(tokenizer, words, [400, 401])
    full = tokenizer(
        words, is_split_into_words=True, add_special_tokens=False, verbose=False
    )
    for index, window in zip([400, 401], windows):
        expected = [
            piece
            for piece, word in zip(full.input_ids, full.word_ids())
            if word == index
        ]
        assert len(expected) > 0
        assert [
            window.input_ids[position] for position in window.target_positions
        ] == expected
        assert len(window.input_ids) <= 256
        assert window.source_start > 0
        assert window.source_end < window.source_length


def test_repeated_word_offsets_and_request_order_are_preserved(tokenizer):
    words = ["bank", "loan"] + ["record"] * 350 + ["river", "bank"]
    late, early, repeated = target_windows(tokenizer, words, [353, 0, 353])
    assert late == repeated
    assert late.source_end == late.source_length
    assert early.source_start == 0
    assert late.input_ids != early.input_ids
    assert late.target_positions[-1] == len(late.input_ids) - 2
    assert early.target_positions == (1,)


def test_invalid_or_unrepresentable_targets_fail_explicitly(tokenizer):
    with pytest.raises(IndexError):
        target_windows(tokenizer, ["word"], [1])
    with pytest.raises(IndexError):
        target_windows(tokenizer, ["word"], [-1])
    with pytest.raises(ValueError, match="no wordpieces"):
        target_windows(tokenizer, [""], [0])
    with pytest.raises(ValueError, match="complete target"):
        target_windows(tokenizer, ["unaffordable"], [0], limit=3)
    assert target_windows(tokenizer, ["word"], []) == []


def test_windowed_sense_is_selected_only_by_explicit_config(monkeypatch):
    monkeypatch.setattr(pipeline, "build_sense", lambda model, config: E.GlossSense())
    monkeypatch.setattr(pipeline, "bert_vectors", lambda path: {})
    config = {"bert_sense": {"vectors": "unused", "temperature": 0.02, "weight": 0.2}}
    normal, normal_round_trip = pipeline.build_senses(None, config)
    assert type(normal.bert) is E.BertSense
    config["bert_sense"]["target_window"] = True
    windowed, windowed_round_trip = pipeline.build_senses(None, config)
    assert type(windowed.bert) is WindowedBertSense
    assert type(normal_round_trip) is type(windowed_round_trip)


def test_batch_order_and_literal_pad_attention_match_unbatched_requests(
    tokenizer, monkeypatch
):
    import torch
    from types import SimpleNamespace
    from cleartext import contextual
    from cleartext.context_window import windowed_word_vectors

    masks = []

    def encoder(input_ids, attention_mask, output_hidden_states):
        masks.append((input_ids.clone(), attention_mask.clone()))
        # Distinct token/position values let the test catch target-owner swaps
        # across the 16-request batch boundary, without loading encoder weights.
        positions = torch.arange(input_ids.shape[1]).expand_as(input_ids).float()
        hidden = torch.stack(
            (input_ids.float() % 17, positions, torch.ones_like(positions)), dim=-1
        )
        return SimpleNamespace(hidden_states=(hidden,) * 4)

    fake = SimpleNamespace(
        bert=encoder, config=SimpleNamespace(max_position_embeddings=512)
    )
    monkeypatch.setattr(contextual, "model", lambda: (tokenizer, fake))
    requests = [(["bank", "[PAD]", "river"], [2, 0]), (["loan"], [])]
    requests += [
        (["the", word, "."], [1])
        for word in [
            "cat",
            "dog",
            "tree",
            "house",
            "car",
            "book",
            "fish",
            "bird",
            "table",
            "chair",
            "road",
            "sky",
            "rain",
            "sun",
            "moon",
            "river",
        ]
    ]
    together = windowed_word_vectors(requests)
    assert [len(group) for group in together] == [
        len(indices) for _, indices in requests
    ]
    first_ids, first_mask = masks[0]
    pad_position = list(first_ids[0]).index(tokenizer.pad_token_id)
    assert first_mask[0, pad_position] == 1
    for (words, indices), group in zip(requests, together):
        for index, vector in zip(indices, group):
            single = windowed_word_vectors([(words, [index])])[0][0]
            np.testing.assert_allclose(vector, single)
