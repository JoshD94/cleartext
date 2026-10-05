import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.features import nlp
from cleartext.generation import WordNetGenerator
from cleartext.refine import features_specific
from cleartext.relations import RELATION_NAMES, features_relations, relation_values


def values(target, pos, candidate):
    return dict(zip(RELATION_NAMES, relation_values(target, pos, candidate)))


def test_relations_distinguish_broadening_from_mixed_verb_relations():
    broad = values("encode", "v", "convert")
    useful = values("commence", "v", "begin")
    assert broad["hypernym_fraction"] == 1
    assert broad["same_sense_fraction"] == 0
    assert broad["hypernym_loss_mean"] > 0
    # begin is a synonym in some senses and linked through a verb group in
    # others. A single generator source flag cannot express this mixture.
    assert useful["same_sense_fraction"] > 0
    assert useful["verb_group_fraction"] > 0
    assert useful["hypernym_fraction"] < 1


def test_similar_adjective_has_no_hypernym_loss():
    row = values("convoluted", "a", "complex")
    assert row["similar_fraction"] == 1
    assert row["hypernym_loss_mean"] == row["hypernym_loss_max"] == 0


def test_cached_and_live_features_match_and_preserve_v7_prefix():
    doc = nlp()("The algorithm encodes a message.")
    token = next(token for token in doc if token.text == "encodes")
    info, generated = WordNetGenerator(
        ("synonym", "hypernym", "similar"), multiword=True, fallback=True
    ).generate(token)
    candidates = [
        {
            **candidate,
            "x": np.ones(15),
            "fit": 0.6,
            "gain_word": 0.1,
            "gain_context": 0.1,
            "target_difficulty": 0.4,
        }
        for candidate in generated
    ]
    live = {
        "target": token.text,
        "text": doc.text,
        "target_info": info,
        "candidates": candidates,
    }
    cached = {
        **live,
        "candidates": [
            {k: v for k, v in candidate.items() if k != "senses"}
            for candidate in candidates
        ],
    }
    live_feature, cached_feature, old_feature = (
        features_relations(live),
        features_relations(cached),
        features_specific(live),
    )
    for first, second in zip(live["candidates"], cached["candidates"]):
        new, old = live_feature(live, first), old_feature(live, first)
        np.testing.assert_array_equal(new, cached_feature(cached, second))
        np.testing.assert_array_equal(new[: len(old)], old)
        assert len(new) == len(old) + len(RELATION_NAMES)
    broad = next(c for c in candidates if c["word"] == "converts")
    full = live_feature(live, broad)
    assert full[-1] == full[-len(RELATION_NAMES) + 4]
    broad["x"][0] = 0.1
    assert live_feature(live, broad)[-1] == pytest.approx(0.1 * full[-1])


def test_unknown_provenance_is_an_error_instead_of_a_neutral_score():
    with pytest.raises(ValueError, match="target_info"):
        features_relations({"candidates": [{"lemma": "convert"}]})
    with pytest.raises(ValueError, match="provenance"):
        relation_values("encode", "v", "pineapple")


def test_cases_without_wordnet_candidates_remain_valid_abstentions():
    assert features_relations({"target_info": None, "candidates": []})({}, {}) is None
