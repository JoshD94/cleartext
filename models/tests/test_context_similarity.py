import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext import context_similarity as C
from cleartext.complete_slot import SlotInput


def test_pooling_uses_all_span_pieces_and_length_masks_literal_padding():
    calls = []

    def encoder(**kwargs):
        calls.append(kwargs)
        ids = kwargs["input_ids"].float()
        vectors = torch.stack((ids, torch.ones_like(ids)), -1)
        return SimpleNamespace(
            hidden_states=(vectors, vectors * 2, vectors * 3, vectors * 4)
        )

    inputs = [
        SlotInput((101, 0, 4, 102), (1, 2), (0, 4)),
        SlotInput((101, 6, 102), (1,), (6,)),
    ]
    vectors = C.slot_vectors(
        inputs, SimpleNamespace(pad_token_id=0), SimpleNamespace(bert=encoder), 2
    )
    np.testing.assert_allclose(vectors[0], np.asarray([2, 1]) / np.sqrt(5), atol=1e-7)
    np.testing.assert_allclose(vectors[1], np.asarray([6, 1]) / np.sqrt(37), atol=1e-7)
    assert calls[0]["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    with pytest.raises(ValueError, match="batch size"):
        C.slot_vectors(inputs, None, None, 0)


def test_similarity_compares_each_replacement_to_original_and_empty_never_loads(
    monkeypatch,
):
    monkeypatch.setattr(
        C.contextual,
        "model",
        lambda: (
            None,
            SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)),
        ),
    )
    calls = []
    monkeypatch.setattr(
        C, "slot_inputs", lambda *args, **kwargs: calls.append((args, kwargs)) or []
    )
    monkeypatch.setattr(
        C, "slot_vectors", lambda *args: np.asarray([[1, 0], [1, 0], [0, 1], [-1, 0]])
    )
    np.testing.assert_array_equal(
        C.replacement_similarities(
            "bank near bank", 10, 14, ["bank", "shore", "opposite"]
        ),
        [1, 0, -1],
    )
    assert calls[0][0][4] == ["bank", "bank", "shore", "opposite"]
    monkeypatch.setattr(
        C.contextual, "model", lambda: pytest.fail("empty request must not load BERT")
    )
    assert C.replacement_similarities("", 0, 0, []).size == 0


def test_cached_feature_prefix_and_missing_candidate_score(monkeypatch):
    monkeypatch.setattr(
        C,
        "features_prompted_relations",
        lambda case: lambda case, candidate: np.arange(51),
    )
    case = {"candidates": [{"word": "shore"}], "replacement_similarity": {"shore": 0.8}}
    row = C.features_prompted_similarity(case)(case, case["candidates"][0])
    np.testing.assert_array_equal(row[:51], np.arange(51))
    assert row[-1] == 0.8
    case["replacement_similarity"] = {}
    with pytest.raises(KeyError):
        C.features_prompted_similarity(case)(case, case["candidates"][0])
    assert C.features_prompted_similarity({"candidates": []})({}, {}) is None


def test_live_span_accepts_capitalization_and_rejects_wrong_offset(monkeypatch):
    monkeypatch.setattr(
        C,
        "features_prompted_relations",
        lambda case: lambda case, candidate: np.zeros(51),
    )
    calls = []
    monkeypatch.setattr(C, "similarity_for", lambda *args: calls.append(args) or (0.8,))
    case = {
        "text": "Bank near bank.",
        "target": "bank",
        "start": 0,
        "candidates": [{"word": "shore"}],
    }
    assert C.features_prompted_similarity(case)(case, case["candidates"][0])[-1] == 0.8
    assert calls == [("Bank near bank.", 0, 4, ("shore",))]
    case["start"] = 2
    with pytest.raises(ValueError, match="exact target span"):
        C.features_prompted_similarity(case)
