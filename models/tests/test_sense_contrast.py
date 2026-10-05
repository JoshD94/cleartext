import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.sense_contrast import (
    SenseContrastScorer,
    sense_connections,
    features_sense_contrast,
)


def scorer(vectors):
    return SenseContrastScorer(
        SimpleNamespace(
            temperature=0.02,
            sense_vector=lambda name: vectors.get(name, np.array([0.0, 1.0])),
        )
    )


def test_sense_connections_distinguish_identity_broadening_and_verb_groups():
    assert sense_connections("deposit.v.02")["deposit.v.02"] == "same"
    assert sense_connections("deposit.v.02")["give.v.03"] == "broader"
    assert sense_connections("encode.v.01")["convert.v.02"] == "broader"
    assert sense_connections("convert.v.02")["convert.v.11"] == "related"


def test_candidate_posterior_exposes_an_unrelated_dominant_sense():
    from nltk.corpus import wordnet as wn

    own = [sense.name() for sense in wn.synsets("bank", pos="n")]
    assert "bank.n.01" in own
    unrelated = next(name for name in own if name != "bank.n.01")
    vectors = {name: np.array([0.0, 1.0]) for name in own}
    vectors[unrelated] = np.array([1.0, 0.0])
    values = scorer(vectors).score(
        np.array([1.0, 0.0]), {"bank.n.01": 1.0}, "n", [{"lemma": "bank"}]
    )[0]
    assert values[0] < 0.01 and values[3] > 0.99 and values[4] < -0.9
    assert np.isclose(values[:4].sum(), 1.0)


def test_hypernym_mass_does_not_become_same_sense_mass():
    values = scorer(
        {"encode.v.01": np.array([1.0, 0.0]), "convert.v.02": np.array([1.0, 0.0])}
    ).score(np.array([1.0, 0.0]), {"encode.v.01": 1.0}, "v", [{"lemma": "convert"}])[0]
    assert values[0] == 0 and values[1] > 0.99 and values[3] < 0.01


def test_absent_senses_and_invalid_inputs_are_explicit():
    model = scorer({})
    assert not model.score(np.array([1.0, 0.0]), {}, "n", [{"lemma": "bank"}])[0].any()
    assert not model.score(
        np.array([1.0, 0.0]), {"bank.n.01": 1.0}, "n", [{"lemma": "xqz_unknown"}]
    )[0].any()
    with pytest.raises(ValueError):
        model.score(np.array([0.0, 0.0]), {}, "n", [])
    with pytest.raises(ValueError):
        model.score(np.array([1.0, 0.0]), {"bank.n.01": -0.1}, "n", [])
    with pytest.raises(ValueError):
        model.score(np.array([1.0, 0.0]), {"bank.n.01": 0.5}, "n", [])
    with pytest.raises(ValueError):
        features_sense_contrast({"candidates": [{"word": "bank"}]})


def test_rewritten_scoring_uses_frozen_fit_and_marks_unscored_candidates(monkeypatch):
    from cleartext import sense_contrast as module

    calls = []

    def vectors(text, start, end, words):
        calls.append(words)
        return [np.array([1.0, 0.0]) for word in words]

    monkeypatch.setattr(module, "replacement_vectors", vectors)
    model = SimpleNamespace(bert=scorer({"bank.n.01": np.array([1.0, 0.0])}).bert)
    case = {
        "text": "The bank is here.",
        "target": "bank",
        "start": 4,
        "target_info": ("bank", "n"),
        "sense_distribution": {"bank.n.01": 1.0},
        "candidates": [
            {"word": "shore", "lemma": "shore", "fit": 0.9},
            {"word": "slope", "lemma": "slope", "fit": 0.1},
            {"word": "banking concern", "lemma": "banking concern", "fit": 0.8},
        ],
    }
    result = module.rewritten_sense_contrast(case, model, limit=2)
    assert calls == [["shore", "banking concern"]]
    assert result["shore"][-1] == 1 and not result["slope"].any()
    assert module.rewritten_sense_contrast({"candidates": []}, model) == {}
    with pytest.raises(ValueError):
        module.rewritten_sense_contrast(case, model, limit=0)
    with pytest.raises(ValueError):
        module.rewritten_sense_contrast({**case, "start": 0}, model)
