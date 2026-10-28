"""Discourse preservation and screened-pair boundaries, using authored text."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.features import nlp
from cleartext.structure.discourse import discourse_checks
from cleartext.structure.quality import audit_rows, select_training, text_key
from cleartext.structure.seq2seq import generation_checks


@pytest.mark.parametrize("marker", ["so", "otherwise", "however"])
def test_losing_discourse_marker_fails_runtime_contract(marker):
    source = f"Alice checked the figures, {marker} Bob revised the detailed report."
    output = "Alice checked the figures. Bob revised the detailed report."
    assert "discourse_markers" in generation_checks(source, output)["failed"]


def test_retaining_causal_connection_can_pass_shared_contract():
    source = "Alice checked the figures, so Bob revised the detailed report."
    output = "Alice checked the figures. So Bob revised the detailed report."
    assert generation_checks(source, output)["pass"]


def test_maybe_must_still_cover_both_shared_subject_predicates():
    source = "Maybe Alice saw the error and did not understand it at the time."
    bad = "Maybe Alice saw the error. Alice did not understand it at the time."
    good = "Maybe Alice saw the error. Maybe Alice did not understand it at the time."
    assert not discourse_checks(nlp()(source), nlp()(bad))["uncertainty_scope"]
    assert discourse_checks(nlp()(source), nlp()(good))["uncertainty_scope"]


def test_new_subject_does_not_automatically_inherit_uncertainty():
    source = "Maybe Alice saw the error and Bob reviewed the detailed report."
    output = "Maybe Alice saw the error. Bob reviewed the detailed report."
    assert discourse_checks(nlp()(source), nlp()(output))["uncertainty_scope"]


def grammar(text, rule=None):
    return {
        "original": text,
        "available": True,
        "suggestions": [] if rule is None else [{"rule": rule, "source": "error"}],
    }


def pair(ident, source, reference):
    return {"id": ident, "source": source, "reference": reference}


def test_screening_quarantines_both_bad_inputs_and_targets_without_relabeling():
    a = "Alice checked the figures and Bob revised the report."
    b = "Alice checked the figures. Bob revised the report."
    c = "Carol reviewed the budget and David prepared the report."
    d = "Carol reviewed the budget. David prepared the report."
    rows = [pair("source_flag", a, b), pair("target_flag", c, d), pair("copy", b, b)]
    cache = {text_key(t): grammar(t) for t in (a, b, c, d)}
    cache[text_key(a)] = grammar(a, "NON3PRS_VERB")
    cache[text_key(d)] = grammar(d, "NON3PRS_VERB")
    retained, quarantine = select_training(rows, cache, [])
    assert retained == [rows[2]]
    assert [q["quarantine_reasons"] for q in quarantine] == [
        ["source:flagged"],
        ["reference:flagged"],
    ]
    assert all(q["reference"] == r["reference"] for q, r in zip(quarantine, rows))


def test_training_cannot_overlap_reserved_source_or_target():
    a = "Alice reviewed the report and Bob checked the figures."
    row = pair("a", a, a)
    with pytest.raises(ValueError, match="overlap"):
        select_training(
            [row], {text_key(a): grammar(a)}, [pair("test", "Other text.", a)]
        )


def test_inherited_grammar_warning_is_not_counted_as_new():
    a = "Alice checked the figures and Bob revised the report."
    b = "Alice checked the figures. Bob revised the report."
    cache = {text_key(t): grammar(t, "NON3PRS_VERB") for t in (a, b)}
    report = audit_rows([{**pair("a", a, b), "output": b}], cache, "output")
    assert report["shared_warning_candidate_inputs"] == 1
    assert report["new_warning_candidate_inputs"] == 0
