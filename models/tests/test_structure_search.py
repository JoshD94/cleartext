"""Candidate selection, using authored grammar fixtures and a fake decoder."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.features import nlp
from cleartext.structure.common import normalize
from cleartext.structure.search import (
    CandidateGenerator,
    boundary_checks,
    repair_boundaries,
    select_candidate,
)

SOURCE = "Alice reviewed the report and Bob checked the figures."
SPLIT = "Alice reviewed the report. Bob checked the figures."


def proposal(text, terminated=True):
    return {"text": text, "terminated": terminated}


@pytest.mark.parametrize(
    "text",
    [
        "The server, which stores customer records, stopped responding.",
        "As the server checks a request, it logs the result.",
        "Alice reviewed the report, and Bob checked the figures.",
        "Alice reviewed the report; Bob checked the figures.",
        'Alice said, "Bob checked the figures, Carol reviewed the report."',
        "Alice reviewed the report, the figures, and the summary.",
    ],
)
def test_valid_or_ambiguous_punctuation_is_left_alone(text):
    assert repair_boundaries(text) == text


def test_repair_preserves_words_and_fixes_independent_comma_boundary():
    text = "Alice reviewed the report, Bob checked the figures."
    repaired = repair_boundaries(text)
    assert repaired == SPLIT
    words = lambda s: [t.lower_ for t in nlp()(s) if not t.is_punct]
    assert words(text) == words(repaired)
    assert boundary_checks(nlp()(SOURCE), nlp()(repaired))["no_comma_splice"]
    assert not boundary_checks(nlp()(SOURCE), nlp()(text))["no_comma_splice"]


def test_leading_subordinate_fragment_cannot_promote_embedded_predicate():
    source = (
        "As we look ahead and set our sights on the Nation we know we can be "
        "and we must be, I thank my predecessors for their presence today."
    )
    output = source.replace(", I thank", ". I thank")
    assert boundary_checks(nlp()(source), nlp()(source))["independent_boundaries"]
    assert not boundary_checks(nlp()(source), nlp()(output))["independent_boundaries"]


def test_valid_copy_wins_over_lower_ranked_rewrite():
    result = select_candidate(SOURCE, [proposal(SOURCE), proposal(SPLIT)])
    assert result["output"] == SOURCE and result["selected_beam"] == 0


def test_invalid_first_beam_does_not_hide_valid_alternative():
    result = select_candidate(
        SOURCE, [proposal("Alice reviewed the report."), proposal(SPLIT)]
    )
    assert result["output"] == SPLIT and result["selected_beam"] == 1
    assert not result["raw_checks"]["pass"]


def test_repair_is_checked_and_can_be_selected():
    result = select_candidate(
        SOURCE, [proposal("Alice reviewed the report, Bob checked the figures.")]
    )
    assert result["output"] == SPLIT
    assert result["selected_origin"] == "boundary_repair"
    assert result["output_checks"]["pass"]


def test_all_failed_candidates_return_original_with_trace():
    result = select_candidate(SOURCE, [proposal("Alice reviewed the report.")])
    assert result["output"] == SOURCE
    assert result["selected_origin"] == "identity_fallback"
    assert result["candidates"][0]["checks"]["failed"]


def test_terminated_duplicate_can_follow_truncated_candidate():
    result = select_candidate(SOURCE, [proposal(SPLIT, False), proposal(SPLIT)])
    assert result["output"] == SPLIT and result["selected_beam"] == 1
    assert "generation_terminated" in result["raw_checks"]["failed"]


def test_decoder_requests_four_candidates_and_checks_each_eos():
    import torch

    class Tokenizer:
        eos_token_id = 1

        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[2, 3]])}

        def decode(self, row, **kwargs):
            return SPLIT if row[1].item() == 4 else "Alice reviewed the report."

    class Model:
        def generate(self, **kwargs):
            assert kwargs["num_beams"] == kwargs["num_return_sequences"] == 4
            assert kwargs["do_sample"] is False
            return torch.tensor([[0, 4, 2], [0, 4, 1], [0, 3, 1], [0, 2, 1]])

    generator = CandidateGenerator.__new__(CandidateGenerator)
    generator.tokenizer, generator.model, generator.device = Tokenizer(), Model(), "cpu"
    result = generator.predict(SOURCE)
    assert normalize(result["output"]) == normalize(SPLIT)
    assert result["selected_beam"] == 1
    assert len(result["model_candidates"]) == 4


def test_fresh_sample_excludes_prior_sources_and_full_targets():
    from cleartext.structure.search_experiment import fresh_test

    rows = [
        {"source": SOURCE, "reference": SPLIT},
        {
            "source": "Carol reviewed the report and Dave checked the figures.",
            "reference": "Carol reviewed the report. Dave checked the figures.",
        },
    ]
    selected = fresh_test(rows, {normalize(SPLIT)}, 1)
    assert selected == [rows[1]]
    with pytest.raises(ValueError, match="eligible fresh"):
        fresh_test(rows, {normalize(SPLIT)}, 2)


def test_fresh_sample_does_not_filter_on_reference_grammar():
    from cleartext.structure.search_experiment import fresh_test

    row = {"source": SOURCE, "reference": "Alice and Bob in the room."}
    assert fresh_test([row], set(), 1) == [row]


def test_evaluation_cannot_overwrite_or_use_changed_protocol(tmp_path):
    import json

    from cleartext.structure.expanded import digest
    from cleartext.structure.search_experiment import evaluate

    output = tmp_path / "results"
    output.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        evaluate(tmp_path, tmp_path, "test", output)
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "data_sha256": digest(tmp_path / "data.json"),
                "settings": {},
            }
        )
    )
    with pytest.raises(ValueError, match="Settings changed"):
        evaluate(tmp_path, tmp_path, "test", tmp_path / "new")
