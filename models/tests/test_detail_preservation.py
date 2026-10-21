import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext import detail_preservation as D


def test_context_mass_distinguishes_same_sense_from_broadening():
    scorer = D.DetailPreservationScorer()
    broad = scorer.evidence({"database.n.01": 1.0}, {"lemma": "information"})
    assert broad["broader"] == 1 and broad["expected_specificity_loss"] > 0
    assert not broad["semantic_certified"]
    same = scorer.evidence({"commence.v.01": 1.0}, {"lemma": "begin"})
    assert same["same_sense"] == 1 and same["broader"] == 0


def test_polysemous_candidate_follows_context_mass_and_keeps_similar_adjectives():
    scorer = D.DetailPreservationScorer()
    compatible = scorer.features(
        {"bank.n.01": 0.8, "depository_financial_institution.n.01": 0.2},
        {"lemma": "banking_company"},
    )
    swapped = scorer.features(
        {"bank.n.01": 0.2, "depository_financial_institution.n.01": 0.8},
        {"lemma": "banking_company"},
    )
    assert compatible[0] == pytest.approx(0.2) and swapped[0] == pytest.approx(0.8)
    assert (
        scorer.evidence({"convoluted.s.02": 1.0}, {"lemma": "complex"})["broader"] == 0
    )
    assert scorer.features({}, {"lemma": "begin"})[-1] == 0
    with pytest.raises(ValueError, match="sum to one"):
        scorer.features({"commence.v.01": 0.2}, {"lemma": "begin"})


def test_new_features_require_context_and_preserve_old_columns(monkeypatch):
    monkeypatch.setattr(
        D,
        "features_prompted_similarity",
        lambda case: lambda case, candidate: np.arange(52),
    )
    case = {
        "candidates": [{"lemma": "begin"}],
        "sense_distribution": {"commence.v.01": 1.0},
    }
    row = D.features_detail_preservation(case)(case, case["candidates"][0])
    np.testing.assert_array_equal(row[:52], np.arange(52))
    assert row.shape == (57,)
    del case["sense_distribution"]
    with pytest.raises(ValueError, match="full contextual"):
        D.features_detail_preservation(case)


def test_new_breadth_feature_clamps_legacy_traversal_overshoot(monkeypatch):
    D.sense_edge.cache_clear()
    monkeypatch.setattr(
        D, "descendants", lambda name: 20002 if name == "information.n.01" else 0
    )
    first = D.sense_edge("database.n.01", "information")[1]
    D.sense_edge.cache_clear()
    monkeypatch.setattr(
        D, "descendants", lambda name: 20012 if name == "information.n.01" else 0
    )
    assert first == D.sense_edge("database.n.01", "information")[1] == np.log1p(20000)
    D.sense_edge.cache_clear()
