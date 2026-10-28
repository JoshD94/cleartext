import numpy as np
import pytest

from cleartext import sense_usage as U


def test_rare_shared_reading_has_lower_prior_than_primary_reading():
    rare = U.usage_evidence("teach", "v", "learn", {"teach.v.01": 1.0}, 0.8)
    primary = U.usage_evidence("teach", "v", "instruct", {"teach.v.01": 1.0}, 0.8)
    assert 0 < rare[0] < primary[0] <= 1
    assert rare[3] == pytest.approx(rare[0] * 0.8)
    # The two lemmas have different frame inventories even in this shared sense.
    assert 0 < rare[4] < 1 and rare[5] == 1


def test_prior_includes_zero_count_senses_and_normalizes():
    prior = dict(U.sense_priors("learn", "v"))
    assert len(prior) > 1 and all(p > 0 for p in prior.values())
    assert sum(prior.values()) == pytest.approx(1)
    assert U.sense_priors("take in", "v") == U.sense_priors("TAKE_IN", "v")


def test_same_word_has_no_shared_prior_gap_and_adjectives_have_no_verb_frames():
    row = U.usage_evidence("convoluted", "a", "convoluted", {"convoluted.s.02": 1}, 1)
    assert row[0] > 0 and row[7] == 0 and row[8] == 1
    assert row[4] == row[5] == 0
    missing = U.usage_evidence("convoluted", "a", "complex", {}, 1)
    assert missing[8] == 0 and missing[0] == missing[1] == 0


@pytest.mark.parametrize("distribution", [
    {"teach.v.01": -1}, {"teach.v.01": float("nan")}, {"teach.v.01": 0.2},
])
def test_invalid_context_distributions_are_rejected(distribution):
    with pytest.raises(ValueError, match="sense probabilities"):
        U.usage_evidence("teach", "v", "learn", distribution, 1)


def test_context_senses_must_belong_to_the_target():
    with pytest.raises(ValueError, match="original target"):
        U.usage_evidence("teach", "v", "learn", {"bank.n.01": 1}, 1)


def test_wordnet_morphological_ambiguity_keeps_both_analyses():
    prior = dict(U.sense_priors("found", "v"))
    assert "establish.v.01" in prior and "find.v.01" in prior
    row = U.usage_evidence("found", "v", "find", {"find.v.01": 1}, 0.5)
    assert row[0] > 0 and row[8] == 1
    same = U.usage_evidence("found", "v", "found", {"find.v.01": 1}, 1)
    assert same[0] > 0 and same[7] == 0
    bounded = U.usage_evidence("bound", "v", "bind", {"bind.v.02": 1}, 0.5)
    assert bounded[0] > 0


def test_new_features_preserve_prefix_and_require_context(monkeypatch):
    monkeypatch.setattr(U, "features_prompted_similarity", lambda case: lambda case, c: np.arange(52))
    c = {"lemma": "instruct", "x": [0.75]}
    case = {"candidates": [c], "target_info": ("teach", "v"),
            "sense_distribution": {"teach.v.01": 1}}
    row = U.features_sense_usage(case)(case, c)
    np.testing.assert_array_equal(row[:52], np.arange(52))
    assert row.shape == (52 + len(U.USAGE_FEATURE_NAMES),)
    del case["sense_distribution"]
    with pytest.raises(ValueError, match="contextual senses"):
        U.features_sense_usage(case)
    assert U.features_sense_usage({"candidates": []})({}, {}) is None
