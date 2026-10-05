import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.ensemble_pipeline import EnsembleClearText
from cleartext.features import nlp
from cleartext.generation import WordNetGenerator
from cleartext.preservation import ConservativeDetailClearText, broader_links


def candidates(text, word):
    doc = nlp()(text)
    token = next(t for t in doc if t.text == word)
    return {
        c["word"]: c
        for c in WordNetGenerator(
            ("synonym", "hypernym", "similar"), multiword=True, fallback=True
        ).generate(token)[1]
    }


def test_detail_policy_catches_broadening_across_domains():
    examples = [
        ("The database stores records.", "database", "information"),
        ("The algorithm encodes a message.", "encodes", "converts"),
        ("The filibuster delayed the vote.", "filibuster", "delay"),
        ("She deposited the money at the bank.", "deposited", "gave"),
    ]
    for text, target, replacement in examples:
        assert broader_links(candidates(text, target)[replacement])


def test_detail_policy_keeps_useful_synonyms_and_similar_adjectives():
    assert not broader_links(
        candidates("The firm will commence the audit.", "commence")["begin"]
    )
    assert not broader_links(
        candidates(
            "Although this sentence is convoluted, the idea is simple.", "convoluted"
        )["complex"]
    )


def test_cached_provenance_matches_live_generator():
    proposal = candidates("The database stores records.", "database")["information"]
    cached = {key: value for key, value in proposal.items() if key != "senses"}
    assert broader_links(cached, ("database", "n", "NN", False)) == broader_links(
        proposal
    )
    with pytest.raises(ValueError, match="target_info is required"):
        broader_links(cached)


def test_filter_falls_back_after_scoring_and_default_stays_unchanged():
    doc = nlp()("The database stores records.")
    proposals = candidates(doc.text, "database")
    broad = {**proposals["information"], "accept": 0.9, "rejections": []}
    # A lower scoring positive proposal remains available after the broad candidate is rejected.
    same = {
        "word": "store",
        "lemma": "store",
        "senses": [],
        "source": "synonym",
        "accept": 0.6,
        "rejections": [],
    }
    default = object.__new__(EnsembleClearText)
    default.decision_threshold = 0.45
    experimental = object.__new__(ConservativeDetailClearText)
    experimental.decision_threshold = 0.45
    assert (
        default.first_guarded(doc, [broad, same], guard=False)["word"] == "information"
    )
    assert (
        experimental.first_guarded(doc, [broad, same], guard=False)["word"] == "store"
    )
    assert "experimental detail policy" in broad["rejections"][0]
