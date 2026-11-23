"""Structure v2 harness: vote, interval, tally and pool helpers (no models)."""

from evals import structure_v2 as S


def test_majority_and_most_severe():
    assert S.majority(["good", "good", "meaning"]) == "good"
    assert S.majority(["good", "notsimpler", "meaning"]) == "meaning"
    assert S.majority(["good", "notsimpler", "ungrammatical"]) == "ungrammatical"


def test_wilson_interval():
    lo, hi = S.wilson(37, 40)
    assert 0.79 < lo < 0.81 and 0.97 < hi < 0.98
    assert S.wilson(0, 0) == [None, None]


def test_tally_counts_shared_cards_per_arm():
    items = [{"arm": "a", "card": "c1", "rules": ["r"], "source": "x"},
             {"arm": "a", "card": "c2", "rules": ["r"], "source": "y"},
             {"arm": "b", "card": "c1", "rules": ["s"], "source": "x"}]
    marks = [{"c1": "good", "c2": "meaning"}, {"c1": "good", "c2": "meaning"}, {"c1": "notsimpler", "c2": "good"}]
    out, agree = S.tally(items, marks, ["a", "b"], {"a": {"changed": 2, "sentences": 100}})
    assert out["a"]["n"] == 2 and out["a"]["good"] == 1 and out["a"]["marks"] == {"good": 1, "meaning": 1}
    assert out["b"]["good_rate"] == 1.0
    assert out["a"]["edits_per_100_sentences"] == 2.0
    assert 0 < agree["pairwise"] < 1


def test_pool_helpers():
    assert S.bucket("rfc9000") == S.bucket("rfc9000")
    assert S.clean("The committee approved the new budget for the coming fiscal year.")
    assert not S.clean("Too short.")
    assert not S.clean("DOSAGE AND ADMINISTRATION The dose is one tablet taken twice a day with food.")
    assert S.grams("a b c d e f g h i", 8) == {tuple("abcdefgh"), tuple("bcdefghi")}
