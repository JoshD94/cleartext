import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.candidate_diagnostics import diagnose_case, review_shortlist


def candidate(word, listed=False, guard=True, sense=0.5):
    return {"word": word, "gold": listed, "guard": guard, "x": [sense]}


@pytest.mark.parametrize(
    "rows,scores,stage,outcome",
    [
        ([], {}, "generation", "abstained"),
        ([candidate("a")], {"a": 0.8}, "generation", "unlisted_edit"),
        ([candidate("a", True, False)], {"a": 0.8}, "guard", "abstained"),
        ([candidate("a", True, sense=0.09)], {"a": 0.8}, "meaning_floor", "abstained"),
        ([candidate("a", True)], {"a": 0.44}, "probability", "abstained"),
        (
            [candidate("a", True), candidate("b")],
            {"a": 0.44, "b": 0.8},
            "probability",
            "unlisted_edit",
        ),
        (
            [candidate("a", True), candidate("b")],
            {"a": 0.7, "b": 0.8},
            "ranking",
            "unlisted_edit",
        ),
        (
            [candidate("a", True, sense=0.10)],
            {"a": 0.45},
            "selected_listed",
            "listed_edit",
        ),
    ],
)
def test_first_blocking_stage_and_outcome(rows, scores, stage, outcome):
    result = diagnose_case({"candidates": rows}, scores)
    assert result["stage"] == stage and result["outcome"] == outcome


def test_equal_probabilities_keep_original_candidate_order():
    result = diagnose_case(
        {"candidates": [candidate("first"), candidate("second", True)]},
        {"first": 0.8, "second": 0.8},
    )
    assert result["stage"] == "ranking" and result["selected"]["word"] == "first"


def test_probability_mismatch_and_nan_are_rejected():
    with pytest.raises(ValueError, match="keys"):
        diagnose_case({"candidates": [candidate("a")]}, {})
    with pytest.raises(ValueError, match="finite"):
        diagnose_case({"candidates": [candidate("a")]}, {"a": float("nan")})


def test_shortlist_ignores_labels_and_deduplicates_by_surface():
    rows = [
        {
            "word": word,
            "lemma": word,
            "source": "bert",
            "senses": [],
            "bert_rank": rank,
            "bert_probability": 0.1,
            "gold": listed,
        }
        for word, rank, listed in [
            ("third", 3, True),
            ("first", 1, False),
            ("FIRST", 2, True),
            ("second", 4, True),
        ]
    ]
    selected = review_shortlist(rows, limit=2)
    assert [row["word"] for row in selected] == ["first", "third"]
    assert all("gold" not in row for row in selected)
    for row in rows:
        row["gold"] = not row["gold"]
    assert review_shortlist(rows, limit=2) == selected
    with pytest.raises(ValueError, match="limit"):
        review_shortlist(rows, limit=0)
