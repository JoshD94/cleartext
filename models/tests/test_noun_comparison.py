"""Keep paired development ranks independent of WordNet enumeration order."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from cleartext.refine import features_listwise
from check_noun_inflection import align_order


def test_enumeration_order_cannot_change_cached_tied_fit_ranks():
    candidates = [
        dict(
            word=word,
            lemma=word,
            source="synonym",
            multiword=False,
            pos_fallback=False,
            x=np.zeros(15),
            fit=0.5,
            gain_word=0.0,
            gain_context=0.0,
            target_difficulty=0.5,
        )
        for word in ("complex", "difficult", "hard")
    ]
    case = {
        "target": "complicated",
        "text": "It was complicated.",
        "candidates": candidates,
    }
    expected = np.stack(
        [features_listwise(case)(case, candidate) for candidate in candidates]
    )
    reversed_case = {**case, "candidates": list(reversed(candidates))}
    shuffled = features_listwise(reversed_case)
    assert not np.array_equal(shuffled(reversed_case, candidates[0]), expected[0])
    aligned = {
        **case,
        "candidates": align_order(reversed_case["candidates"], candidates),
    }
    actual = np.stack(
        [features_listwise(aligned)(aligned, candidate) for candidate in candidates]
    )
    np.testing.assert_array_equal(actual, expected)
    renamed = [
        {**candidates[1], "word": "difficult phrase"},
        candidates[0],
        candidates[2],
    ]
    assert [candidate["lemma"] for candidate in align_order(renamed, candidates)] == [
        candidate["lemma"] for candidate in candidates
    ]
