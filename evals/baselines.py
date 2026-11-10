"""Baseline systems for ClearText evaluation.

Per Section 3 of the proposal, every system variant is compared against:
  1. The original text (identity baseline).
  2. A simple rule-based simplifier that replaces difficult words with
     dictionary-based substitutions.

Each baseline is a callable: ``simplified = baseline(original_text)``.
"""

import json
import re
from pathlib import Path

_DEFAULT_DICT = Path(__file__).with_name("data") / "simplification_dict.json"


class OriginalTextBaseline:
    """Identity baseline: returns the input unchanged."""

    name = "original"

    def __call__(self, text):
        return text


class RuleBasedSimplifier:
    """Dictionary-based substitution baseline.

    Replaces whole-word occurrences of complex terms with simpler
    equivalents, case-insensitively, preserving the original
    capitalization of the first letter.
    """

    name = "rule_based"

    def __init__(self, dictionary=None):
        if dictionary is None:
            with open(_DEFAULT_DICT, encoding="utf-8") as f:
                dictionary = json.load(f)
        # Longest keys first so multi-word entries win over their parts.
        self._entries = sorted(dictionary.items(), key=lambda kv: -len(kv[0]))
        self._patterns = [
            (re.compile(r"\b" + re.escape(src) + r"\b", re.IGNORECASE), tgt)
            for src, tgt in self._entries
        ]

    def __call__(self, text):
        out = text
        for pattern, replacement in self._patterns:
            def _sub(m, replacement=replacement):
                word = m.group(0)
                if word[:1].isupper():
                    return replacement[:1].upper() + replacement[1:]
                return replacement

            out = pattern.sub(_sub, out)
        return out
