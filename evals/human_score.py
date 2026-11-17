"""Kea-Roy's human-calibrated simplicity score S as a passage metric.

S(original, simplified) = G*S_lex + H*S_read + I*S_jargon, the Week 1 formula
with the weights `train_weights.py` fitted to human judgments and saved in
`trained_weights.json`: word weights A-E from CompLex (held-out r 0.68) and
sentence weights G/H/I from SimpEval (max-correlation fit, held-out r 0.60).
That is the recommended configuration in evals/README.md and WEIGHTS_REPORT.md:
the GBM word model wins at the word level, but sentence weights refit on its
S_lex were worse (r 0.54), so S stays linear-based.

The features are train_weights.sentence_features, restated here so importing
this module does not put evals/ on sys.path (evals/datasets.py would shadow
the Hugging Face `datasets` package). test_testset_v3.py checks parity.

S is relative: it scores how much simpler the output is than its source.
Higher is simpler. An unchanged text scores H*0.5 (S_read = 0.5, no lexical
or jargon reduction). SimpEval rated single sentences; here S is computed per
passage, which is outside the fitted range and is reported as such.
"""

import json
from pathlib import Path

from wordfreq import zipf_frequency

from .evaluate_base import clamp, jargon_simplicity, lexical_simplicity, readability_score, words

WEIGHTS = Path(__file__).resolve().parent / "trained_weights.json"
_weights = None


def trained_weights():
    global _weights
    if _weights is None:
        raw = json.loads(WEIGHTS.read_text())
        _weights = {"word": raw["word_weights"], "sentence": raw["sentence_weights"]}
    return _weights


def features(original, simplified, word_weights=None):
    """S_lex, S_read, S_jargon exactly as in train_weights.sentence_features."""
    word_weights = word_weights or trained_weights()["word"]
    before, after = words(original), words(simplified)
    technical_terms = {w for w in before if zipf_frequency(w, "en") < 3.0}
    return {
        "S_lex": lexical_simplicity(before, after, word_weights),
        "S_read": clamp(readability_score(simplified) - readability_score(original) + 0.5),
        "S_jargon": jargon_simplicity(before, after, technical_terms),
    }


def simplicity(original, simplified):
    """{S_lex, S_read, S_jargon, S} with the trained weights."""
    weights = trained_weights()["sentence"]
    out = features(original, simplified)
    out["S"] = weights["G"] * out["S_lex"] + weights["H"] * out["S_read"] + weights["I"] * out["S_jargon"]
    return out
