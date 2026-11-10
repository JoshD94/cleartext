import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext import prompted_specificity as P


def test_interactions_preserve_features_and_separate_relation_support(monkeypatch):
    values = np.zeros(51)
    values[[39, 40, 43, 48, 49]] = [0.25, 0.75, 2.0, 1.0, 0.5]
    monkeypatch.setattr(
        P, "features_prompted_relations", lambda case: lambda case, candidate: values
    )
    feature = P.features_prompted_specificity({})
    supported = feature({}, {})
    np.testing.assert_array_equal(supported[:51], values)
    np.testing.assert_array_equal(supported[51:], [0.125, 0.375, 0.0, 1.0])
    values[48:51] = 0.0
    np.testing.assert_array_equal(feature({}, {})[51:], [0.0, 0.0, 0.75, 0.0])


def test_empty_candidate_case_does_not_infer():
    assert P.features_prompted_specificity({"candidates": []})({}, {}) is None
