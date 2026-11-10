import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext import prompted_relations as P


def test_new_features_preserve_existing_rows_and_use_surface_form_support(monkeypatch):
    monkeypatch.setattr(
        P, "features_relations", lambda case: lambda case, candidate: np.arange(48)
    )
    case = {
        "candidates": [{"word": "Complex"}, {"word": "involved"}],
        "bert_support": {"complex": (2, 0.25)},
    }
    feature = P.features_prompted_relations(case)
    np.testing.assert_array_equal(
        feature(case, case["candidates"][0])[:48], np.arange(48)
    )
    np.testing.assert_array_equal(
        feature(case, case["candidates"][0])[-3:], [1, 0.5, 0.25]
    )
    np.testing.assert_array_equal(feature(case, case["candidates"][1])[-3:], [0, 0, 0])


def test_live_support_passes_exact_offset_and_empty_cases_never_infer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        P, "features_relations", lambda case: lambda case, candidate: np.zeros(48)
    )
    monkeypatch.setattr(P, "support_for", lambda *args: calls.append(args) or {})
    case = {
        "text": "bank near bank",
        "target": "bank",
        "start": 10,
        "target_info": ("bank", "n", "NN", False),
        "candidates": [{"word": "shore"}],
    }
    P.features_prompted_relations(case)
    assert calls == [("bank near bank", "bank", 10, ("bank", "n", "NN", False))]
    P.features_prompted_relations({"candidates": []})
    assert len(calls) == 1
    del case["start"]
    with pytest.raises(ValueError, match="exact target offset"):
        P.features_prompted_relations(case)
