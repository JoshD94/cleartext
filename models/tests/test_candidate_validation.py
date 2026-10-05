import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.bert_candidates import proposals
from cleartext.candidate_validation import (
    exact_lemma_senses,
    validate_proposal,
    validated_proposals,
)
from cleartext.features import nlp
from cleartext.proposal_evidence import wordnet_evidence


def test_exact_lemma_does_not_borrow_irregular_verb_senses():
    found = {sense.name() for sense in exact_lemma_senses("found", "v")}
    assert "establish.v.01" in found and "detect.v.01" not in found
    saw = {sense.name() for sense in exact_lemma_senses("saw", "v")}
    assert "saw.v.01" in saw and "witness.v.02" not in saw
    assert not exact_lemma_senses("consisted", "v")
    evidence = wordnet_evidence({"detect.v.01": 1.0}, "v", "found")
    assert evidence["source_support_by_relation"]["same"] == 0.0


def test_dictionary_validation_rejects_double_inflection_and_keeps_irregular_forms():
    assert not validate_proposal({"lemma": "consisted", "word": "consisteds"}, "v")[
        "dictionary_valid"
    ]
    assert validate_proposal({"lemma": "find", "word": "found"}, "v")[
        "dictionary_valid"
    ]
    assert validate_proposal({"lemma": "child", "word": "children"}, "n")[
        "dictionary_valid"
    ]
    assert not validate_proposal({"lemma": "find", "word": "founded"}, "v")[
        "dictionary_valid"
    ]
    with pytest.raises(ValueError, match="unsupported"):
        validate_proposal({"lemma": "find", "word": "found"}, "x")


def test_opt_in_filter_preserves_existing_proposals_and_empty_origins():
    token = next(
        token
        for token in nlp()("The device consists of several parts.")
        if token.text == "consists"
    )
    predictions = [
        {"token": word, "rank": index + 1, "probability": 0.1}
        for index, word in enumerate(["consisted", "contain", "include"])
    ]
    target, original = proposals(token, predictions)
    assert any(candidate["word"] == "consisteds" for candidate in original)
    kept = validated_proposals(original, target[1])
    assert "consisteds" not in {candidate["word"] for candidate in kept}
    assert {"contains", "includes"} <= {candidate["word"] for candidate in kept}
    assert any(candidate["word"] == "consisteds" for candidate in original)
    for candidate in kept:
        assert candidate in original
        assert candidate["senses"] == next(
            row["senses"] for row in original if row["word"] == candidate["word"]
        )
