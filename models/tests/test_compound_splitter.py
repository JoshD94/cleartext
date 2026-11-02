"""Authored mechanism regressions, not held-out semantic quality labels."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.adapter import rewrite
from cleartext.structure.compound import DependencyCompoundSplitter


@pytest.fixture(scope="module")
def splitter():
    return DependencyCompoundSplitter()


@pytest.mark.parametrize("source,expected", [
    ("The store closed early, and the workers went home.",
     "The store closed early. The workers went home."),
    ("The store closed early and the workers went home.",
     "The store closed early. The workers went home."),
    ("The store closed early, but the workers stayed outside.",
     "The store closed early. But the workers stayed outside."),
    ("The store closed early but the workers stayed outside.",
     "The store closed early. But the workers stayed outside."),
    ("The store closed early, so the workers went home.",
     "The store closed early. So the workers went home."),
    ("The server checks each request and sends a response to the client.",
     "The server checks each request. The server sends a response to the client."),
    ("Alice read the report and checked the figures.",
     "Alice read the report. Alice checked the figures."),
    ("The neural network processed 500 records, and the analyst checked the results.",
     "The neural network processed 500 records. The analyst checked the results."),
    ("The server checks each request but sends a response.",
     "The server checks each request. But the server sends a response."),
    ("The report was reviewed, and the figures were checked.",
     "The report was reviewed. The figures were checked."),
])
def test_actual_parser_splits_and_retains_source_details(splitter, source, expected):
    result = splitter.predict(source)
    assert result["output"] == expected, result
    assert result["checks"]["pass"]
    assert not result["semantic_certified"]
    assert result["edits"]


@pytest.mark.parametrize("source", [
    "Alice bought bread and milk.",
    "Alice and Bob packed the bags.",
    "Alice opened the door and Bob.",
    "If the store closes early, Alice leaves and Bob stays.",
    "Alice said that Bob left and Carol stayed.",
    "Alice hopes to leave and Bob stays.",
    "The server, which stores records, checks requests and sends responses.",
    "The server may check requests and send responses.",
    "The store closed, but the workers did not leave.",
    "Either Alice leaves or Bob stays.",
    "Alice left so that Bob could stay.",
    "Alice worked so hard that Bob stayed.",
    "A scientist reviewed the report and checked the figures.",
    "Someone reviewed the report and checked the figures.",
    "The server has checked the request and sent the response.",
    "The server carefully checks requests and sends responses.",
    "Alice bought and read the book.",
    "Alice bought the book and read.",
    # Known parser failure: QUIC is tagged as a participle. Keep the input.
    "The gateway uses Diffie-Hellman keys, and the client stores QUIC packets.",
    "The gateway uses Diffie-Hellman keys, and the client stores each response.",
    "The server checks requests but sends responses.",
    # Known parser failure: the main predicate is mistaken for a noun phrase.
    "The server, which stores records, checks requests, and the client displays results.",
    "In 2025, Alice left and Bob stayed.",
    'Alice said "Bob left and Carol stayed."',
    "The server checks `a and b` and sends the response.",
    "The store closed (early), and the workers went home.",
    "The store closed early and the workers went home?",
])
def test_ambiguous_scope_fragments_and_literals_abstain(splitter, source):
    assert splitter.predict(source)["output"] == source


def test_passages_preserve_whitespace_and_exceed_four_sentences(splitter):
    source = "The store closed early, and the workers went home.\n\n" + "Bob waited. " * 5
    result = splitter.predict(source)
    assert result["output"] == source.replace("early, and the", "early. The")
    output, edits, details = rewrite(source, splitter)
    assert output == result["output"]
    assert edits and details["rule"] == "dependency compound split"


def test_rollback_when_final_checks_fail(splitter, monkeypatch):
    original = splitter.validate

    def reject(source, output):
        result = original(source, output)
        result.update({"pass": False, "failed": ["authored rejection probe"]})
        return result

    monkeypatch.setattr(splitter, "validate", reject)
    source = "The store closed early, and the workers went home."
    result = splitter.predict(source)
    assert result["output"] == source
    assert result["accepted_count"] == 0
    assert all(not d["accepted"] for d in result["boundaries"])


def test_import_and_predict_do_not_load_generative_libraries(splitter):
    before = set(sys.modules)
    splitter.predict("The store closed early, and the workers went home.")
    added = set(sys.modules) - before
    assert not any(name.split(".")[0] in {"torch", "transformers", "sentence_transformers"} for name in added)


def test_eval_variants_keep_old_ladder_and_use_local_splitter(monkeypatch):
    from evals import model_adapter, runner

    class Pipeline:
        def analyze(self, text, **kwargs):
            assert kwargs["structure"] and kwargs["phrases"] and kwargs["words"]
            assert kwargs["guardrails"] is True
            assert kwargs["structure_rewriter"].backend == "dependency_compound"
            return kwargs["structure_rewriter"].predict(text)

    monkeypatch.setattr(model_adapter, "_load_pipe", lambda run: Pipeline())
    monkeypatch.setattr(runner, "SYSTEMS", dict(runner.SYSTEMS))
    original_toggles = {name: dict(kw) for name, kw in model_adapter._ABLATION_KWARGS.items()}
    model_adapter.register_compound_systems(model_adapter.COMPOUND_SYSTEMS)
    text = "The store closed early, and the workers went home."
    expected = "The store closed early. The workers went home."
    for name in model_adapter.COMPOUND_SYSTEMS:
        assert runner.SYSTEMS[name](text) == expected
    assert model_adapter._ABLATION_KWARGS == original_toggles


@pytest.mark.parametrize("source,expected", [
    ("Alice, who wrote the report, checked the figures, and Bob signed the document.",
     "Alice, who wrote the report, checked the figures. Bob signed the document."),
    ("Alice checked the report, and the analyst who wrote it reviewed the figures.",
     "Alice checked the report. The analyst who wrote it reviewed the figures."),
    ("Alice, who wrote and revised the report, checked the figures, and Bob signed the document.",
     "Alice, who wrote and revised the report, checked the figures. Bob signed the document."),
])
def test_relative_clause_stays_with_its_noun(splitter, source, expected):
    result = splitter.predict(source)
    assert result["output"] == expected, result
    assert result["checks"]["checks"]["relative_attachments"]
    assert result["accepted_count"] == 1
    assert result["boundaries"][-1]["reason"] == "accepted"


@pytest.mark.parametrize("source", [
    "The analyst who checked the report and reviewed the figures left.",
    "Alice said that the server stores records, and the client displays results.",
    "If Alice checks the report, Bob reviews the figures, and Carol writes the summary.",
    "The server, which stores records, checks requests and sends responses.",
])
def test_embedded_coordination_and_shared_scope_still_abstain(splitter, source):
    result = splitter.predict(source)
    assert result["output"] == source, result
    assert result["accepted_count"] == 0
    assert all(d["reason"] and not d["accepted"] for d in result["boundaries"])


def test_joint_subject_guard_rejects_separate_events(splitter):
    checks = splitter.validate("Alice and Bob lifted the piano.",
                               "Alice lifted the piano. Bob lifted the piano.")
    assert not checks["pass"]
    assert "subject_groups" in checks["failed"]
    assert not splitter.validate("Alice praised Bob.", "Bob praised Alice.")["pass"]
    assert not splitter.validate("The server may check requests and send responses.",
                                 "The server may check requests. The server sends responses.")["pass"]


def test_length_policy_gates_original_clause_lengths():
    source = "The store closed early, and the workers went home."
    result = DependencyCompoundSplitter(min_sentence_words=20).predict(source)
    assert result["output"] == source
    assert result["boundaries"][0]["reason"] == "sentence_too_short"
    result = DependencyCompoundSplitter(min_clause_words=5).predict(source)
    assert result["output"] == source
    assert result["boundaries"][0]["reason"] == "clause_too_short"
    assert result["boundaries"][0]["left_words"] == 4
    assert result["boundaries"][0]["right_words"] == 4
    result = DependencyCompoundSplitter(9, 4).predict(source)
    assert result["output"] != source
    assert result["length_policy"] == {"min_sentence_words": 9, "min_clause_words": 4}
    # Copied subjects do not inflate the original right-clause length.
    result = DependencyCompoundSplitter(min_clause_words=5).predict(
        "The server checks each request and sends a response.")
    assert result["accepted_count"] == 0


@pytest.mark.parametrize("kwargs", [
    {"min_sentence_words": -1}, {"min_clause_words": 1.5},
    {"min_clause_words": True}, {"min_sentence_words": "20"},
])
def test_invalid_length_policy(kwargs):
    with pytest.raises(ValueError, match="nonnegative integer"):
        DependencyCompoundSplitter(**kwargs)


def test_diagnostics_separate_detection_from_acceptance(splitter):
    source = "Alice bought bread and milk.\nThe store closed early, and the workers went home."
    result = splitter.predict(source)
    assert result["candidate_count"] == 2
    assert result["accepted_count"] == 1
    rejected, accepted = result["boundaries"]
    assert rejected["reason"] == "not_matrix_coordination"
    assert accepted["accepted"]
    for boundary in result["boundaries"]:
        assert source[boundary["offset"]:].startswith(boundary["connector"])
    assert "score" not in result  # Checks/lengths are not calibrated confidence.


def test_passage_rollback_clears_boundary_acceptance(monkeypatch):
    model = DependencyCompoundSplitter()
    validate = model.validate

    def reject_passage(source, output):
        checks = validate(source, output)
        if "\n" in source and source != output:
            # Override only the final passage check, after a sentence edit passed.
            checks["pass"] = False
            checks["failed"] = ["authored_passage_rejection"]
        return checks

    monkeypatch.setattr(model, "validate", reject_passage)
    source = "The store closed early, and the workers went home.\nBob waited."
    result = model.predict(source)
    assert result["raw"] != source
    assert result["output"] == source
    assert not result["edits"] and result["accepted_count"] == 0
    assert result["boundaries"][0]["reason"] == "passage_rollback"
    assert not result["boundaries"][0]["accepted"]


@pytest.mark.parametrize("fragment", ["Note: archived.", "(2) Includes archived records."])
def test_unchanged_fragment_does_not_veto_complete_split(splitter, fragment):
    source = "The store closed early, and the workers went home.\n\n" + fragment
    result = splitter.predict(source)
    assert result["output"] == "The store closed early. The workers went home.\n\n" + fragment
    assert result["checks"]["pass"]
    assert result["accepted_count"] == 1
    assert result["boundaries"][0]["accepted"]


@pytest.mark.parametrize("source,output", [
    ("Alice opened the door and Bob.", "Alice opened the door. And Bob."),
    ("Alice waited. Note: archived.", "Alice waited. Note: changed."),
    ("Alice waited. Note: archived.", "Alice waited. Note: archived. Note: archived."),
])
def test_new_or_modified_fragments_still_fail_validation(splitter, source, output):
    result = splitter.validate(source, output)
    assert not result["pass"]
    assert not result["checks"]["complete_clauses"]
