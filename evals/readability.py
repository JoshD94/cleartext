"""Readability and simplicity metrics for the ClearText evaluation pipeline.

Implements the "Readability and Simplicity" dimension from Section 3.1 of the
ClearText project proposal:
  - Flesch Reading Ease (higher = easier)
  - Flesch-Kincaid Grade Level (lower = easier)
  - Gunning Fog Index (lower = easier)
  - SMOG Index (lower = easier)
  - Average sentence length, average word length
  - Technical term count (heuristic: rare words via wordfreq)

A successful simplification should increase reading ease and reduce
grade-level scores, sentence length, and technical terminology.

Implementation note: syllable counting and the four classic formulas are
implemented directly (standard vowel-group heuristic) so the pipeline has
no data-download dependencies at runtime.
"""

import math
import re

import textstat

try:
    from wordfreq import zipf_frequency

    _HAS_WORDFREQ = True
except ImportError:  # optional dependency
    _HAS_WORDFREQ = False

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_VOWELS = set("aeiouy")


def syllable_count(word):
    """Heuristic syllable count: vowel groups, silent-e and -es/-ed handling."""
    w = word.lower().strip()
    if not w:
        return 0
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    if w.endswith(("es", "ed")) and count > 1 and len(w) > 4:
        # crude: don't double-penalize; vowel-group pass already handles most
        pass
    return max(count, 1)


def _counts(text):
    """Shared (words, sentences, syllables, complex_words) counts."""
    words = _WORD_RE.findall(text)
    n_words = len(words)
    try:
        n_sent = textstat.sentence_count(text) or 0
    except Exception:
        n_sent = 0
    if n_sent == 0 and n_words > 0:
        n_sent = 1
    syllables = sum(syllable_count(w) for w in words)
    complex_words = sum(1 for w in words if syllable_count(w) >= 3)
    return n_words, n_sent, syllables, complex_words


def flesch_reading_ease(text):
    """0-100 scale; higher means easier to read."""
    n_words, n_sent, syllables, _ = _counts(text)
    if not n_words or not n_sent:
        return float("nan")
    return 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / n_words)


def flesch_kincaid_grade(text):
    """US school grade level; lower means easier."""
    n_words, n_sent, syllables, _ = _counts(text)
    if not n_words or not n_sent:
        return float("nan")
    return 0.39 * (n_words / n_sent) + 11.8 * (syllables / n_words) - 15.59


def gunning_fog(text):
    """Years of education needed; lower means easier."""
    n_words, n_sent, _, complex_words = _counts(text)
    if not n_words or not n_sent:
        return float("nan")
    return 0.4 * ((n_words / n_sent) + 100 * (complex_words / n_words))


def smog_index(text):
    """Grade-level estimate from polysyllabic words; lower means easier."""
    n_words, n_sent, _, complex_words = _counts(text)
    if not n_sent:
        return float("nan")
    # Standard SMOG formula; reasonable approximation for short texts.
    return 1.0430 * math.sqrt(complex_words * (30 / n_sent)) + 3.1291


def word_tokens(text):
    return _WORD_RE.findall(text)


def avg_sentence_length(text):
    """Mean words per sentence."""
    n_words, n_sent, _, _ = _counts(text)
    if not n_sent:
        return float("nan")
    return n_words / n_sent


def avg_word_length(text):
    """Mean characters per alphabetic token."""
    words = word_tokens(text)
    if not words:
        return float("nan")
    return sum(len(w) for w in words) / len(words)


def technical_term_count(text, zipf_threshold=3.5):
    """Heuristic count of jargon-like tokens.

    With `wordfreq` installed: counts alphabetic tokens longer than 4
    characters whose Zipf frequency is below `zipf_threshold` (rare words).
    Without it: falls back to counting tokens with 3+ syllables.
    """
    words = [w.lower() for w in word_tokens(text)]
    if not words:
        return 0
    if _HAS_WORDFREQ:
        return sum(
            1
            for w in words
            if len(w) > 4 and zipf_frequency(w, "en") < zipf_threshold
        )
    return sum(1 for w in words if syllable_count(w) >= 3)


def readability_report(text):
    """Compute the full readability metric suite for one passage."""
    return {
        "flesch_reading_ease": flesch_reading_ease(text),
        "flesch_kincaid_grade": flesch_kincaid_grade(text),
        "gunning_fog": gunning_fog(text),
        "smog_index": smog_index(text),
        "avg_sentence_length": avg_sentence_length(text),
        "avg_word_length": avg_word_length(text),
        "technical_term_count": technical_term_count(text),
    }
