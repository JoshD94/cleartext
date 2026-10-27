"""Data screening must not mistake style warnings or missing checks for bad grammar."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.grammar_screening import manifest, screen_result


def saved(*rules, available=True):
    return {"available": available, "suggestions": [{"rule": r} for r in rules]}


def test_style_and_dialect_preferences_do_not_disqualify_grammatical_text():
    result = screen_result(saved("EXTREME_ADJECTIVES", "WHETHER", "DIFFERENT_THAN"))
    assert result["status"] == "no_covered_issue"
    assert not result["grammar_certified"]
    assert len(result["ignored_rules"]) == 3


def test_grammar_error_is_flagged_even_if_no_correction_was_applied():
    grammar = saved("NON3PRS_VERB", "COMMA_COMPOUND_SENTENCE")
    grammar["suggestions"][0].update(
        applied=False, reason="unsupported_or_ambiguous_edit"
    )
    result = screen_result(grammar)
    assert result["status"] == "flagged"
    assert result["error_rules"] == ["NON3PRS_VERB"]


def test_spelling_and_unknown_rules_require_review_instead_of_automatic_exclusion():
    result = screen_result(saved("MORFOLOGIK_RULE_EN_US", "FUTURE_RULE"))
    assert result["status"] == "review" and not result["error_rules"]


def test_quoted_code_or_detected_name_warning_is_not_a_grammar_failure():
    grammar = saved("EN_CONTRACTION_SPELLING")
    grammar["suggestions"][0]["reason"] = "protected_span_or_insertion"
    assert screen_result(grammar)["status"] == "no_covered_issue"


@pytest.mark.parametrize(
    "grammar",
    [
        saved(available=False),
        {**saved(), "reason": "Invalid grammar matches: ValueError"},
        {"available": True},
        {"available": True, "suggestions": None},
        {"available": True, "suggestions": [{}]},
        {"available": True, "suggestions": [{"rule": ""}]},
        {"available": True, "suggestions": [None]},
    ],
)
def test_failed_or_malformed_check_is_never_counted_as_clean(grammar):
    assert screen_result(grammar)["status"] == "unavailable"


def test_manifest_retains_all_ids_order_and_original_text_without_rewriting():
    rows = [
        {
            "id": "a",
            "source": "They is ready.",
            "grammar": {
                **saved("PERS_PRONOUN_AGREEMENT"),
                "original": "They is ready.",
            },
        },
        {
            "id": "b",
            "source": "An opaque input.",
            "grammar": {**saved(available=False), "original": "An opaque input."},
        },
        {
            "id": "c",
            "source": "Looks fine.",
            "grammar": {**saved(), "original": "Looks fine."},
        },
    ]
    before = [r["source"] for r in rows]
    result = manifest(rows, "frozen-source")
    assert result["input_n"] == result["retained_n"] == 3
    assert [r["id"] for r in result["records"]] == ["a", "b", "c"]
    assert [r["source"] for r in rows] == before
    assert result["status_counts"] == {
        "flagged": 1,
        "unavailable": 1,
        "no_covered_issue": 1,
    }


def test_manifest_rejects_warnings_scored_against_a_different_input():
    row = {
        "id": "a",
        "source": "They is ready.",
        "grammar": {**saved(), "original": "They are ready."},
    }
    with pytest.raises(ValueError, match="unchanged original"):
        manifest([row], "frozen-source")
