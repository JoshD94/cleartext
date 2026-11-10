"""Repeated reporting subjects and shared train/inference eligibility."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.features import nlp
from cleartext.structure.expanded import reference_quality
from cleartext.structure.search import checked
from cleartext.structure.seq2seq import generation_checks


@pytest.mark.parametrize("subject", ["Alice", "She", "We"])
def test_independent_report_with_repeated_matrix_subject_can_split(subject):
    verb, hope = ("think", "hope") if subject == "We" else ("thinks", "hopes")
    source = f"{subject} {verb} that Bob works here and {subject.lower() if subject != 'Alice' else subject} {hope} that Carol will join him."
    output = f"{subject} {verb} that Bob works here. {subject} {hope} that Carol will join him."
    result = generation_checks(source, output)
    assert result["pass"], result["failed"]
    assert (
        reference_quality(
            {"source": source, "reference": output}, nlp()(source), nlp()(output)
        )
        is None
    )


@pytest.mark.parametrize(
    "source,output",
    [
        (
            "Alice thinks that Bob works here and Bob hopes that Carol will join him.",
            "Alice thinks that Bob works here. Bob hopes that Carol will join him.",
        ),
        (
            "Alice said that Bob left the office and Carol stayed at the station.",
            "Alice said that Bob left the office. Carol stayed at the station.",
        ),
        (
            "Alice thinks that Bob works here and that Alice hopes that Carol will join him.",
            "Alice thinks that Bob works here. Alice hopes that Carol will join him.",
        ),
        (
            "Alice thinks that Bob believes that Alice works here and Alice hopes that Carol will join her.",
            "Alice thinks that Bob believes that Alice works here. Alice hopes that Carol will join her.",
        ),
        (
            "Alice thinks that Bob works here and she hopes that Carol will join him.",
            "Alice thinks that Bob works here. She hopes that Carol will join him.",
        ),
    ],
)
def test_embedded_or_unresolved_reporting_scope_remains_conservative(source, output):
    result = generation_checks(source, output)
    assert "complement_scope" in result["failed"]
    assert (
        reference_quality(
            {"source": source, "reference": output}, nlp()(source), nlp()(output)
        )
        is not None
    )


def test_training_and_inference_use_the_same_checks_for_detached_attribution():
    source = "Alice thinks that Bob works near the railway station."
    output = "Alice thinks. Bob works near the railway station."
    source_doc, output_doc = nlp()(source), nlp()(output)
    training = reference_quality(
        {"source": source, "reference": output}, source_doc, output_doc
    )
    inference = generation_checks(source, output, source_doc, output_doc)
    assert training == "+".join(inference["failed"]) == "complement_scope"
    assert checked(source, output, source_doc, boundaries=False) == inference


def test_repeated_passive_auxiliary_uses_runtime_tense_contract_in_training():
    source = "The detailed report was prepared and approved by Alice yesterday."
    output = "The detailed report was prepared by Alice yesterday. The detailed report was approved by Alice yesterday."
    result = generation_checks(source, output)
    reason = reference_quality(
        {"source": source, "reference": output}, nlp()(source), nlp()(output)
    )
    assert result["pass"]
    assert reason is None


def test_normalized_copy_cannot_bypass_shared_checks(monkeypatch):
    from cleartext.structure import expanded

    source = "Alice reviewed the report and Bob checked the figures."
    reference = "Alice reviewed the report and  Bob checked the figures."
    monkeypatch.setattr(
        expanded,
        "generation_checks",
        lambda *args: {"pass": False, "failed": ["argument_roles"]},
    )
    assert (
        reference_quality(
            {"source": source, "reference": reference}, nlp()(source), nlp()(reference)
        )
        == "argument_roles"
    )


def test_dev_sampling_keeps_reference_failures_instead_of_resampling():
    from cleartext.structure.alignment import sample_unseen

    rows = [
        {
            "id": "bad",
            "source": "Alice reviewed the report and Bob checked the figures.",
            "reference": "Bob checked only the figures.",
        }
    ]
    assert sample_unseen(rows, set(), 1) == rows


def test_selection_rewards_splitting_instead_of_punctuation_only_changes():
    from cleartext.structure.alignment import structural_score
    from cleartext.structure.common import sari

    source = "Alice reviewed the report and Bob checked the figures."
    split = "Alice reviewed the report. Bob checked the figures."
    row = {"source": source, "reference": split}
    punctuation_only = "Alice reviewed the report, and Bob checked the figures."
    assert structural_score(row, punctuation_only) == sari(source, source, split)
    assert structural_score(row, split) > structural_score(row, punctuation_only)
