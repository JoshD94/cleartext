"""Offline checks for the window swap-veto study harness (no models loaded)."""

import json

import pytest

from evals import swap_veto as V


def test_choice_prefers_precision_then_balanced_accuracy():
    def g(w, t, ba, prec):
        return {"window": w, "threshold": t, "pooled": {"balanced_accuracy": ba, "precision": prec}}

    grid = [g(6, 0.9, 0.80, 0.85), g(4, 0.97, 0.75, 0.91), g(2, 0.99, 0.70, 0.95), g(None, 0.97, 0.78, 0.92)]
    choice = V.choose(grid)
    assert (choice["window"], choice["threshold"], choice["precision_constraint_met"]) == (None, 0.97, True)
    choice = V.choose([g(6, 0.9, 0.80, 0.85), g(4, 0.97, 0.75, 0.8)])
    assert (choice["window"], choice["precision_constraint_met"]) == (6, False)
    # Ties go to higher precision, then the smaller window.
    assert V.choose([g(6, 0.9, 0.8, 0.9), g(4, 0.9, 0.8, 0.9)])["window"] == 4


def test_confusion_counts_accepted_good_as_true_positives():
    c = V.confusion([True, True, False, False], [True, False, True, False])
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    assert c["precision"] == c["recall"] == c["balanced_accuracy"] == 0.5


def test_gold_pairs_are_both_directions_and_deduplicated():
    rows = [{"before": "a b c d", "after": "a b x d"}]
    pairs = V.gold_pairs(rows, windows=(1, 6, None))
    assert ("b c d", "b x d") in pairs and ("b x d", "b c d") in pairs
    assert ("a b c d", "a b x d") in pairs and len(pairs) == 4  # 6 and None give the full text


def test_design_names_mark_window_veto_arms(tmp_path):
    base = {"words": "v7", "phrases": "authored", "structure": "t5+compound_tagger"}
    design = tmp_path / "d.json"
    design.write_text(json.dumps({"configs": [{"base": {**base, "words": "off"}, "label": "F2"},
                                              {"base": base, "veto": "window", "label": "F2+v7"}]}))
    names = V.design_names(design)
    assert names[1].endswith("@window") and "words=v7" in names[1] and not names[0].endswith("@window")
    assert V.design_labels(design)[names[1]] == "F2+v7"
    design.write_text(json.dumps({"configs": [{"base": {**base, "words": "contextual_veto"}, "veto": "window"}]}))
    with pytest.raises(ValueError):
        V.design_names(design)


def test_lexical_row_credits_a_simpler_word():
    source = "The committee will scrutinize the proposal carefully."
    simpler = "The committee will check the proposal carefully."
    same = V.lexical_row(source, source)
    lex = V.lexical_row(source, simpler)
    assert same["changed_zipf_gain"] is None
    assert lex["word_complexity"] < same["word_complexity"]
    assert lex["rare_share"] < same["rare_share"]
    assert lex["changed_zipf_gain"] > 0
