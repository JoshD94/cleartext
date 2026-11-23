"""Surface realization after a word swap (round 21).

The swap decision is unchanged; this only repairs the indefinite article in
front of the replacement ("an incorrect" -> "a wrong", "a important" -> "an
important"). Five rated swaps in rounds 3-20 kept a mismatched article, and
all five were marked Changes meaning.
"""

import re

_ARTICLE = re.compile(r"(?<![A-Za-z])(a|an|A|An|AN)(\s+)$")
# Spelling-to-sound exceptions for the indefinite article.
_AN_SILENT_H = ("hour", "honest", "honor", "honour", "heir")
_A_VOWEL_LETTER = ("uni", "use", "usu", "uti", "ute", "ubiq", "uran", "uri", "uro", "eu", "ewe", "one", "once")


def wants_an(word):
    """Whether `word` takes "an" (by its first sound, approximated from its spelling)."""
    w = word.lower()
    if not w[:1].isalpha():
        return False
    if w.startswith(_AN_SILENT_H):
        return True
    if w.startswith(_A_VOWEL_LETTER):
        return False
    return w[0] in "aeiou"


def fix_article(after, start, rep):
    """`after` with the indefinite article right before position `start` (where `rep` begins)
    agreeing with `rep`; unchanged when no article precedes it or it already agrees."""
    m = _ARTICLE.search(after[:start])
    if not m:
        return after
    art, gap = m.group(1), m.group(2)
    need = "an" if wants_an(rep) else "a"
    if art.lower() == need:
        return after
    new = need.upper() if art.isupper() and len(art) > 1 else (need.capitalize() if art[0].isupper() else need)
    return after[:m.start(1)] + new + gap + after[start:]
