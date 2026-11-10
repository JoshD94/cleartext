#!/usr/bin/env python3
"""Trained GBM word-complexity model (options C+D).

C: gradient-boosted C(w,s) instead of the linear-sigmoid form. Still
   non-LLM: 6 transparent features, no embeddings.
D: features add Brysbaert concreteness + Kuperman age-of-acquisition
   (see lexicons.py) to the base 4 (neg-Zipf, length, syllables, context).

The model is trained by train_word_model.py and saved to
weights/word_gbm.joblib (+ weights/word_gbm_meta.json). This module is the
only place that featurizes for / scores with the GBM, so training and
serving can never disagree.

API:
    word_complexity_gbm(word, context_words=()) -> float in [0,1]
    lexical_simplicity_gbm(before_words, after_words) -> float in [0,1]
    featurize6(token, sentence_words) -> 6 features (for training)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from joblib import load as joblib_load  # noqa: E402
from wordfreq import zipf_frequency  # noqa: E402

from evaluate_base import (  # noqa: E402
    COMPLEXITY_THRESHOLD,
    normalized_reduction,
    syllables,
    words,
)
from lexicons import aoa, concreteness  # noqa: E402

WEIGHTS_DIR = HERE / "weights"
FEATURES = ["neg_zipf", "word_len", "syllables", "context", "concreteness", "aoa"]

_model = None
_meta = None


def context_complexity(token: str, sentence_words: list[str]) -> float:
    others = [w for w in sentence_words if w != token]
    if not others:
        return 0.0
    return sum(len(w) + syllables(w) for w in others) / len(others)


def featurize6(token: str, sentence_words: list[str]) -> list[float]:
    """The 6 GBM features. Single source of truth for train and serve."""
    t = token.casefold()
    return [
        -zipf_frequency(t, "en"),
        float(len(t)),
        float(syllables(t)),
        context_complexity(t, [w.casefold() for w in sentence_words]),
        concreteness(t),
        aoa(t),
    ]


def _ensure():
    global _model, _meta
    if _model is None:
        _model = joblib_load(WEIGHTS_DIR / "word_gbm.joblib")
        _meta = json.loads((WEIGHTS_DIR / "word_gbm_meta.json").read_text())
    return _model, _meta


def word_complexity_gbm(word: str, context_words=()) -> float:
    """GBM C(w,s): predicted human complexity in [0,1]."""
    model, _ = _ensure()
    feats = featurize6(word, list(context_words))
    return float(np.clip(model.predict([feats])[0], 0.0, 1.0))


def lexical_simplicity_gbm(
    before_words: list[str],
    after_words: list[str],
    threshold: float = COMPLEXITY_THRESHOLD,
) -> float:
    """Threshold-based S_lex using GBM scores (mirrors S_jargon form).

    NOTE: with the well-calibrated GBM, 0.5 is too high an operating point
    (most news words score below it, so counts collapse to ~0). Tuned on
    Stage B train: threshold=0.3 (train r=0.38 / test r=0.40 vs 0.13/-0.06
    at 0.5). Pass threshold explicitly; the default keeps the formula's
    0.5 convention.
    """
    before = sum(word_complexity_gbm(w, before_words) > threshold for w in before_words)
    after = sum(word_complexity_gbm(w, after_words) > threshold for w in after_words)
    return normalized_reduction(before, after)


if __name__ == "__main__":
    s = "The photosynthesis process was explained with abstruse terminology."
    ws = words(s)
    for w in ["photosynthesis", "abstruse", "the"]:
        print(f"{w:16s} GBM C(w,s) = {word_complexity_gbm(w, ws):.3f}")
