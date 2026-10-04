#!/usr/bin/env python3
"""Psycholinguistic lexicons for ClearText eval word features.

- Concreteness: Brysbaert et al. (2014), 40k lemmas, 1-5 scale.
  (Downloaded as the 0-100 rescaled text version; divided by 20 here.)
- Age of acquisition: Kuperman et al. (2012) via the wos-data grade table
  (grade 0-12 ~= mean AoA in years minus 5); converted back to years.

Files live in data/lexicons/ (not in git). OOV words fall back to the
lexicon median. Both are plain lookup tables -- no model, no network at
scoring time.
"""

from __future__ import annotations

import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEX_DIR = HERE / "data" / "lexicons"

_cache: dict[str, dict[str, float]] = {}
_medians: dict[str, float] = {}


def _load(name: str) -> dict[str, float]:
    if name in _cache:
        return _cache[name]
    table: dict[str, float] = {}
    if name == "concreteness":
        path = LEX_DIR / "concreteness_brysbaert_rescaled.txt"
        with path.open(encoding="utf-8") as fh:
            next(fh)  # Symbol Type Concreteness
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                word, _typ, val = parts
                try:
                    # Rescaled file is 0-100 via (x-1)/4*100; invert to 1-5.
                    table[word.casefold()] = 1.0 + float(val) / 25.0
                except ValueError:
                    continue
    elif name == "aoa":
        path = LEX_DIR / "aoa_kuperman_grades.tsv"
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                word, grade = parts
                try:
                    table[word.casefold()] = float(grade) + 5.0  # grade -> years
                except ValueError:
                    continue
    else:
        raise ValueError(f"unknown lexicon {name!r}")
    _cache[name] = table
    _medians[name] = statistics.median(table.values())
    return table


def _deinflect(word: str) -> list[str]:
    """Candidate base forms for an inflected word, longest-first.

    Only candidates actually present in the lexicon are ever used, so
    aggressive stripping is safe ('bias' -> 'bia' is rejected).
    """
    cands: list[str] = []
    if word.endswith("ies") and len(word) > 4:
        cands.append(word[:-3] + "y")  # tries -> try
    for suf in ("es", "s", "ed", "ing", "ly"):
        if word.endswith(suf) and len(word) > len(suf) + 2:
            stem = word[: -len(suf)]
            cands.append(stem)
            if suf in ("ed", "ing"):
                if not stem.endswith("e"):
                    cands.append(stem + "e")  # baked -> bake
                if len(stem) >= 2 and stem[-1] == stem[-2]:
                    cands.append(stem[:-1])  # stopped -> stop
    # de-dupe, keep order
    return list(dict.fromkeys(cands))


def _find(name: str, word: str) -> tuple[float, bool]:
    """(value, found) — found is False when falling back to the median."""
    table = _load(name)
    w = word.casefold()
    if w in table:
        return table[w], True
    for cand in _deinflect(w):
        if cand in table:
            return table[cand], True
    return _median(name), False


def _lookup(name: str, word: str) -> float:
    return _find(name, word)[0]


def _median(name: str) -> float:
    _load(name)
    return _medians[name]


def concreteness(word: str) -> float:
    """Brysbaert concreteness, 1 (abstract) to 5 (concrete)."""
    return _lookup("concreteness", word)


def aoa(word: str) -> float:
    """Age of acquisition in years (Kuperman et al.)."""
    return _lookup("aoa", word)


def coverage(words: list[str]) -> dict[str, float]:
    """Fraction of words found in each lexicon (exact or de-inflected)."""
    out = {}
    for name in ("concreteness", "aoa"):
        hits = sum(1 for w in words if _find(name, w)[1])
        out[name] = hits / max(1, len(words))
    return out


if __name__ == "__main__":
    for w in ["dog", "justice", "photosynthesis", "the", "xyzzy"]:
        print(f"{w:16s} concreteness={concreteness(w):.2f}  aoa={aoa(w):.2f}")
    print("medians:", {k: round(_median(k), 2) for k in ("concreteness", "aoa")})
