"""The default scorer must not allocate sequence-by-vocabulary logits."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext import contextual


def test_masked_head_keeps_legacy_piece_cap_masks_and_batch_boundaries(monkeypatch):
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    batches, projected = [], []

    def encoder(input_ids, attention_mask):
        batches.append((input_ids.clone(), attention_mask.clone()))
        return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))

    def head(hidden):
        # Reject projection of the whole sequence, even when token scores would
        # otherwise be correct. Every projected vector must be a masked slot.
        assert hidden.ndim == 2
        assert torch.all(hidden[:, 0] == tokenizer.mask_token_id)
        projected.append(len(hidden))
        return torch.zeros((len(hidden), tokenizer.vocab_size))

    class Model:
        bert = staticmethod(encoder)
        cls = staticmethod(head)

        def __call__(self, **kwargs):
            raise AssertionError("Full-sequence vocabulary projection is unnecessary")

    monkeypatch.setattr(contextual, "model", lambda: (tokenizer, Model()))
    words = ["one two three four five six seven eight"] + ["small"] * 35
    scores = contextual.slot_scores("[PAD] target", 6, 12, words)

    assert projected == [32, 9]  # Six legacy pieces plus 35 one-piece words.
    assert all(torch.all(mask[:, 1] == 0) for _, mask in batches)
    np.testing.assert_allclose(scores, -np.log(tokenizer.vocab_size), atol=1e-6)
