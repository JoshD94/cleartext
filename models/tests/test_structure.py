"""Structural experiments: alignment, held-out separation, and real failure mechanisms.

Authored examples below are unit fixtures, never experiment training/evaluation gold.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
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
