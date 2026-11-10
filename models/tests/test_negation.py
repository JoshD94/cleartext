"""Negation rule: "not un-" adjectives, the doc's plain pair, double negatives.

Authored examples below are unit fixtures, never experiment training/evaluation gold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.rewrites import negation_rewrite


def check(text, expected, edits=1):
    out, e = negation_rewrite(text)
    assert out == expected, f"{text!r} -> {out!r}, expected {expected!r}"
    assert len(e) == edits, f"{text!r}: {len(e)} edits, expected {edits}"
    return out, e


def test_doc_examples():
    check("It is not uncommon.", "It is common.")
    check("Not an uncommon behavior.", "A common behavior.")
    check("That is not an uncommon behavior.", "That is a common behavior.")
    check("It is not hard.", "It is easy.")


def test_prefix_classes():
    check("It is not impossible.", "It is possible.")
    check("The answer was not incorrect.", "The answer was correct.")
    check("The schedule is not irregular.", "The schedule is regular.")
    check("That is not illegal.", "That is legal.")
    check("The results were not inaccurate.", "The results were accurate.")
    check("That was not an unusual event.", "That was a usual event.")


def test_contraction():
    check("It isn't uncommon.", "It is common.")


def test_double_negatives():
    check("It is not nothing.", "It is something.")
    check("He is not nobody.", "He is somebody.")
    check("There is not no reason to leave.", "There is some reason to leave.")


def test_multi_clause_and_multi_sentence():
    check(
        "It is not uncommon, and the workers went home.",
        "It is common, and the workers went home.",
    )
    check(
        "It is not uncommon. It is not impossible.",
        "It is common. It is possible.",
        edits=2,
    )
    # One edit per sentence: the second negation in the same sentence stays.
    check(
        "It is not unimportant, and it is not unlikely.",
        "It is important, and it is not unlikely.",
    )


def test_attributive_litotes():
    check("The not uncommon answer surprised us.", "The common answer surprised us.")


def test_abstains_on_scope():
    # Negation is over the verb; the adjective sits in the object.
    check("She did not give an uncommon answer.", "She did not give an uncommon answer.", edits=0)


def test_abstains_on_correlatives_and_degree():
    check("Not only was it hard, but it was long.", "Not only was it hard, but it was long.", edits=0)
    check("It is not only uncommon but also rare.", "It is not only uncommon but also rare.", edits=0)
    check("It is not very uncommon.", "It is not very uncommon.", edits=0)
    check("Not all answers are uncommon.", "Not all answers are uncommon.", edits=0)
    check("It is not uncommon, but it is rare.", "It is not uncommon, but it is rare.", edits=0)


def test_abstains_without_a_rule():
    check("The cookies were baked by her.", "The cookies were baked by her.", edits=0)
    check("It is not good.", "It is not good.", edits=0)
    # "not never" has no safe flattening (always vs sometimes), so it stays.
    check("I will not never go.", "I will not never go.", edits=0)


def test_abstains_on_quotes_and_code():
    check('"It is not uncommon," she said.', '"It is not uncommon," she said.', edits=0)
    check("The `not uncommon` value is set.", "The `not uncommon` value is set.", edits=0)


def test_edit_records():
    out, e = negation_rewrite("It is not uncommon.")
    assert e[0]["stage"] == "negation"
    assert e[0]["rule"] == "not-un-prefix"
    assert e[0]["before"] == "It is not uncommon."
    assert e[0]["after"] == "It is common."
    _, e = negation_rewrite("It is not nothing.")
    assert e[0]["rule"] == "double-negative"
