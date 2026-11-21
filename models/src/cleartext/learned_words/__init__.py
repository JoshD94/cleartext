"""Learned word simplification (November 18, 2026).

Four trained stages, no LLM and no rules-only decision:

1. ``eligibility``: a complex-word classifier (CWI 2018) and a term-protection
   classifier (public technical glossaries vs. words with attested simpler
   substitutes), combined into "hard and safe to replace".
2. ``candidates``: WordNet (the v7 generator's relations), masked BERT
   (LSBert-style paired input) and the plainlanguage.gov word list.
3. ``ranker``: a gradient-boosted candidate ranker trained on public lexical
   simplification gold, plus an explicit no-swap gate on the top candidate.
4. ``checker``: a per-edit checker for word swaps (meaning, trained on rated
   SWORDS substitutes and LS gold) and one for sentence splits.

Raw datasets live outside the repository (``CLEARTEXT_LEXSIMP_RAW``); see
``models/docs/LEARNED_WORDS.md``.
"""
