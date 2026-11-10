import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.coverage_policy import adoption, check_group_separation, choose


def test_selection_keeps_meaning_floor_guards_threshold_and_generator_ties():
    rows = [{"word": word, "guard": guard, "x": [sense], "gold": word == "a"}
            for word, guard, sense in [("bad_guard", False, 1), ("bad_sense", True, .099),
                                       ("a", True, .1), ("b", True, .1)]]
    cases = [{"id": "case", "candidates": rows}]
    scores = {("case", word): p for word, p in [("bad_guard", 1), ("bad_sense", 1), ("a", .8), ("b", .8)]}
    metrics, selected = choose(cases, scores, .8)
    assert selected["case"]["word"] == "a" and metrics["precision"] == 1
    assert choose(cases, scores, .80001)[0]["edits"] == 0


def repeats():
    return [{"seed": seed, "default": {"baseline": {"net": 10}, "repaired": {"net": 15}},
             "strict": {"repaired": {"precision": .90, "coverage": .20}}} for seed in range(3)]


def test_adoption_requires_strict_bar_in_every_repeat():
    rows = repeats()
    assert adoption(rows)["adoption_bar_passed"]
    rows[2]["strict"]["repaired"]["precision"] = .89999
    assert not adoption(rows)["adoption_bar_passed"]


def test_adoption_rejects_small_gain_missing_repeats_and_nan():
    rows = repeats()
    rows[0]["default"]["repaired"]["net"] = 14
    assert not adoption(rows)["adoption_bar_passed"]
    rows = repeats()
    rows[0]["strict"]["repaired"]["precision"] = float("nan")
    assert not adoption(rows)["adoption_bar_passed"]
    with pytest.raises(ValueError):
        adoption(rows[:2])


def test_repaired_rows_require_both_target_and_full_context_separation():
    check_group_separation(["tap"], ["A different context."], ["scraping"], ["The observed context."])
    with pytest.raises(ValueError, match="target"):
        check_group_separation(["scraping"], ["Different."], ["scraping"], ["Other."])
    with pytest.raises(ValueError, match="context"):
        check_group_separation(["tap"], [" The OBSERVED\ncontext. "], ["scraping"], ["The observed context."])
