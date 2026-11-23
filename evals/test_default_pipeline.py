"""Default pipeline wiring (no models loaded)."""

from cleartext.structure_v2 import KEPT_V3
from evals import default_pipeline as D


def test_default_structure_is_kept_v3():
    assert D.STRUCTURE_OPS == tuple(KEPT_V3)
    assert D.MIN_ENTAILMENT == 0.5 and D.WORDS_ARM == "p50"
    assert '"t5.min_entailment":0.5' in D.F2 and D.F2.startswith("words=off")


def test_splice_keeps_spacing_and_untouched_sentences():
    text = "The cat sat.  It was utilized daily.\nThen it left."
    swap = {"sentence": 1, "before": "It was utilized daily.", "after": "It was used daily.",
            "rep": "used", "word": "utilized"}
    assert D.splice(text, [swap]) == "The cat sat.  It was used daily.\nThen it left."
    assert D.splice(text, []) == text


def test_splice_refuses_mismatched_sentence():
    swap = {"sentence": 0, "before": "Other.", "after": "Else.", "rep": "Else", "word": "Other"}
    try:
        D.splice("The cat sat. It left.", [swap])
    except ValueError:
        return
    raise AssertionError("mismatch not refused")
