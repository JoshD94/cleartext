"""Structural experiments: alignment, held-out separation, and real failure mechanisms.

Authored examples below are unit fixtures, never experiment training/evaluation gold.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

pytest.importorskip(
    "sacremoses", reason="Optional structural experiments are not installed"
)
from cleartext.features import nlp
from cleartext.structure.common import candidate_pool, normalize, sari, structure_checks
from cleartext.structure.corpus import pair_keys, read_pair, select_pairs
from cleartext.structure.learning import edit_alignment, render_tags, token_features


def test_sari_operations_and_identity():
    # Each n-gram order 1..4 has additions, keeps and deletions, all correct.
    assert sari("a b c d e", "b c d e f", "b c d e f") == pytest.approx(100.0)
    assert sari("a b c d e", "a b c d e", "b c d e f") < 100.0
    assert sari("", "", "") == 0.0


def test_checks_allow_splits_without_token_index_comparison():
    old = "Alice reviewed the report, and Bob checked the figures."
    new = "Alice reviewed the report. Bob checked the figures."
    assert structure_checks(old, new)["pass"]


def test_role_swap_and_lost_detail_are_rejected():
    assert (
        "argument_roles"
        in structure_checks("Alice defeated Bob.", "Bob defeated Alice.")["failed"]
    )
    assert (
        "content_retained"
        in structure_checks(
            "The scientist carefully reviewed the report.",
            "The scientist reviewed the report.",
        )["failed"]
    )


def test_negation_modality_numbers_and_technical_terms():
    old = "The neural network may not process 500 records."
    check = structure_checks(old, "The nerve network will process 50 records.")
    assert {"numbers", "negation", "modality", "technical_phrases"} <= set(
        check["failed"]
    )


def test_shared_modal_scope_cannot_disappear_after_splitting():
    old = "The server may check each request and send a response."
    bad = "The server may check each request. The server sends a response."
    check = structure_checks(old, bad)
    assert check["checks"]["modality"]  # Word counts alone miss this.
    assert not check["checks"]["operator_scope"]


def test_identity_preserves_bytes_in_tagger():
    text = "  The server works.\n"
    doc = nlp()(text)
    assert render_tags(doc, [["KEEP", []] for _ in range(len(doc) + 1)]) == text


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "The server checks each request and sends a reply.",
            "The server checks each request. The server sends a reply.",
        ),
        ("The server checks requests.", "The server checks requests. It works."),
        (
            "The server checks requests, and the client waits.",
            "The server checks requests. The client waits.",
        ),
    ],
)
def test_reference_alignment_round_trip_including_eos(source, target):
    doc = nlp()(source)
    tags = edit_alignment(doc, target)
    assert len(tags) == len(token_features(doc)) == len(doc) + 1
    assert normalize(render_tags(doc, tags)) == normalize(target)


def test_dynamic_subject_copy_is_not_a_memorized_name():
    doc = nlp()("The server checks each request and sends a reply.")
    tags = edit_alignment(
        doc, "The server checks each request. The server sends a reply."
    )
    assert "<SUBJECT>" in json.dumps(tags)
    other = nlp()("The client checks each request and sends a reply.")
    assert "The client sends a reply" in render_tags(other, tags)


def test_bad_tag_length_fails_instead_of_dropping_suffix():
    with pytest.raises(ValueError, match="EOS"):
        render_tags(nlp()("The server works."), [])


def test_pair_selection_excludes_cross_split_sources_and_targets():
    line = "The server checks each new request and the client sends each new response .\tThe server checks each new request . <::::> The client sends each new response .\n"
    row = read_pair(line, "test", 0)
    assert row is not None
    rows, counts, _ = select_pairs([line], "train", 10, pair_keys(row))
    assert not rows and counts["cross_split_overlap"] == 1
    assert select_pairs([line, line], "train", 10)[1]["within_split_duplicate"] == 1


def test_candidate_pool_always_includes_identity_and_is_bounded():
    source = "The server checks each request, and the client sends each response."
    rows = candidate_pool(source)
    assert rows[0]["output"] == source
    assert 1 < len(rows) <= 26
    assert len({normalize(r["output"]) for r in rows}) == len(rows)


def test_mac_warning_stops_own_training(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from cleartext.structure import seq2seq

    monkeypatch.delenv("CLEARTEXT_WAIT_FOR_CAPACITY", raising=False)

    monkeypatch.setattr(seq2seq.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        seq2seq.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3)
    )
    monkeypatch.setattr(
        seq2seq.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="2\n")
    )
    with pytest.raises(RuntimeError, match="pressure is not normal"):
        seq2seq.capacity_check(tmp_path)


def test_mac_warning_can_pause_and_resume_without_losing_training_state(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from cleartext.structure import seq2seq

    monkeypatch.setenv("CLEARTEXT_WAIT_FOR_CAPACITY", "1")
    monkeypatch.setattr(seq2seq.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        seq2seq.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3)
    )
    replies = iter(["2\n", "1\n", "System-wide memory free percentage: 50%"])
    monkeypatch.setattr(
        seq2seq.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=next(replies))
    )
    sleeps = []
    monkeypatch.setattr(seq2seq.time, "sleep", sleeps.append)
    seq2seq.capacity_check(tmp_path)
    assert sleeps == [30]
    assert (
        json.loads((tmp_path / "capacity_status.json").read_text())["state"]
        == "running"
    )


@pytest.fixture
def tiny_t5():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import (
        PreTrainedTokenizerFast,
        T5Config,
        T5ForConditionalGeneration,
    )

    torch.set_num_threads(2)
    vocab = {
        word: i
        for i, word in enumerate(
            ["<pad>", "</s>", "<unk>", "The", "server", "works", ".", "client", "waits"]
        )
    }
    core = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = Whitespace()
    core.post_processor = TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 1)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )
    rows = [
        {"source": "The server works.", "reference": "The server works."},
        {
            "source": "The client waits.",
            "reference": "The client waits. The server works.",
        },
    ]
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=len(vocab),
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
    )
    return model, tokenizer, rows


def test_seq2seq_batch_padding_supports_a_real_gradient_step(tiny_t5):
    import torch
    from cleartext.structure.seq2seq import _batch

    model, tokenizer, rows = tiny_t5
    batch = _batch(tokenizer, rows, torch.device("cpu"))
    assert (batch["labels"] == -100).any()
    assert not (batch["labels"] == tokenizer.pad_token_id).any()
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.shared.weight.grad.abs().sum() > 0


def test_accumulated_training_saves_a_reloadable_checkpoint(
    tiny_t5, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import huggingface_hub
    import torch
    from cleartext.structure import seq2seq

    model, tokenizer, rows = tiny_t5
    initial_weight = model.shared.weight.detach().clone()
    pretrained = tmp_path / "pretrained"
    model.save_pretrained(pretrained)
    tokenizer.save_pretrained(pretrained)
    monkeypatch.setattr(
        huggingface_hub, "model_info", lambda *a, **k: SimpleNamespace(sha="local-test")
    )
    monkeypatch.setattr(seq2seq, "capacity_check", lambda _: None)
    # Stop before the epoch ends, so an unprocessed next batch cannot count.
    data = {"train": rows * 3, "dev": rows}
    run = tmp_path / "run"
    run.mkdir()
    result = seq2seq.train_seq2seq(
        data,
        run,
        max_steps=2,
        batch_size=1,
        accumulation_steps=2,
        model_name=str(pretrained),
        device="cpu",
    )
    assert result["steps_completed"] == 2
    assert result["effective_batch_size"] == 2
    assert result["training_examples_seen"] == 4
    assert 0 <= result["elapsed_seconds"] < 120
    loaded = seq2seq.SentenceGenerator(run / "seq2seq", device="cpu")
    assert not torch.equal(initial_weight, loaded.model.shared.weight)
