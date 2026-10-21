import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.proposal_evidence import wordnet_evidence


def test_broader_and_narrower_are_distinct_and_do_not_certify_meaning():
    broader = wordnet_evidence({"dog.n.01": 1.0}, "n", "canine")
    assert broader["source_support_by_relation"]["broader"] == 1.0
    assert broader["source_support_by_relation"]["same"] == 0.0
    narrower = wordnet_evidence({"canine.n.02": 1.0}, "n", "dog")
    assert narrower["source_support_by_relation"]["narrower"] == 1.0
    assert narrower["semantic_certified"] is False


def test_antonym_has_real_inventory_but_no_fabricated_same_sense():
    evidence = wordnet_evidence({"increase.v.01": 1.0}, "v", "decrease")
    assert evidence["candidate_senses_available"]
    assert evidence["source_support_by_relation"]["antonym"] == 1.0
    assert evidence["source_support_by_relation"]["same"] == 0.0
    assert all(link["source_sense"] == "increase.v.01" for link in evidence["links"])


def test_missing_inventory_is_missing_evidence():
    evidence = wordnet_evidence({}, "n", "qwertyzzzz")
    assert (
        not evidence["source_senses_available"]
        and not evidence["candidate_senses_available"]
    )
    assert all(value == 0 for value in evidence["source_support_by_relation"].values())
    with pytest.raises(ValueError, match="normalized"):
        wordnet_evidence({"dog.n.01": 0.2}, "n", "dog")
