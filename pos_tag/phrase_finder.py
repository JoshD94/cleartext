"""
Complex phrase finder for the sentence simplification project.

Finds wordy or complex phrases (e.g. "to what extent", "in order to",
"made a decision") and suggests simpler replacements.

Setup: same as pos_tag.py (spaCy + en_core_web_sm). This file must be
in the same folder as pos_tag.py, because it reuses the loaded model.

Phrase lists are read from manual_phrases.csv and (if present)
simpleppdb_phrases.csv in the same folder. See build_phrase_list.py.

Usage from your teammate's code:
    from phrase_finder import find_complex_phrases
    phrases = find_complex_phrases("To what extent is this true?")
"""

import csv
from pathlib import Path

from spacy.matcher import Matcher, PhraseMatcher
from spacy.util import filter_spans

from pos_tag import nlp  # reuse the model that pos_tag already loaded

# ---------------------------------------------------------------------------
# Phrase lists (loaded from CSV files in the same folder as this script)
#
#   manual_phrases.csv     - hand-written phrases (edit this to add your own)
#   simpleppdb_phrases.csv - phrases from the SimplePPDB++ dataset, created by
#                            running build_phrase_list.py (optional)
#
# If a phrase appears in both, the manual entry wins.
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PHRASE_FILES = [
    ("manual", HERE / "manual_phrases.csv"),
    ("simpleppdb", HERE / "simpleppdb_phrases.csv"),
]


def _normalize(text):
    """Turn a phrase into the same form spaCy produces, e.g. "Don't" -> "do n't"."""
    return " ".join(t.lower_ for t in nlp.make_doc(text))


def _load_phrases():
    """Read all phrase files into {normalized complex phrase: (suggestion, source)}."""
    phrases = {}
    for source, path in PHRASE_FILES:
        if not path.exists():
            continue
        count = 0
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                complex_p = _normalize(row["complex"].strip())
                simple_p = row["simple"].strip()
                if complex_p and simple_p and complex_p not in phrases:
                    phrases[complex_p] = (simple_p, source)
                    count += 1
        print(f"[phrase_finder] loaded {count:,} phrases from {path.name}")
    return phrases


FIXED_PHRASES = _load_phrases()

# Flexible patterns: match any form of a verb ("make", "makes", "made", ...).
# Format: (name, suggestion, spaCy Matcher pattern)
# The suggestion is the base form; use the phrase's "first_tag" to inflect it.
FLEXIBLE_PATTERNS = [
    ("make a decision", "decide",
     [{"LEMMA": "make"}, {"POS": "DET", "OP": "?"}, {"LOWER": {"IN": ["decision", "decisions"]}}]),
    ("take into account", "consider",
     [{"LEMMA": "take"}, {"LOWER": "into"}, {"LOWER": {"IN": ["account", "consideration"]}}]),
    ("give consideration to", "consider",
     [{"LEMMA": "give"}, {"LOWER": "consideration"}, {"LOWER": "to"}]),
    ("be able to", "can",
     [{"LEMMA": "be"}, {"LOWER": "able"}, {"LOWER": "to"}]),
    ("be in possession of", "have",
     [{"LEMMA": "be"}, {"LOWER": "in"}, {"LOWER": "possession"}, {"LOWER": "of"}]),
    ("come to a conclusion", "conclude",
     [{"LEMMA": "come"}, {"LOWER": "to"}, {"POS": "DET", "OP": "?"}, {"LOWER": "conclusion"}]),
    ("carry out", "do",
     [{"LEMMA": "carry"}, {"LOWER": "out"}]),
]

# ---------------------------------------------------------------------------
# "in a ___ way" -> "___ly"
# A general rule: "in a theatrical way" -> "theatrically",
# "in decisive ways" -> "decisively", "in a very careful manner" -> "very carefully".
# The suggestion is built from the adjective, so one rule covers many phrases.
# ---------------------------------------------------------------------------

WAY_PATTERN_NAME = "in a ___ way"
WAY_PATTERN = [
    {"LOWER": "in"},
    {"LOWER": {"IN": ["a", "an"]}, "OP": "?"},
    {"POS": "ADV", "OP": "?"},                       # optional "very", "highly", ...
    {"POS": "ADJ"},
    {"LOWER": {"IN": ["way", "ways", "manner", "fashion"]}},
]

# Adjectives with an irregular adverb, or whose "-ly" form means something else
ADVERB_EXCEPTIONS = {"good": "well", "public": "publicly", "whole": "wholly"}
NO_ADVERB = {"new", "big", "small", "old", "similar", "same", "certain", "real",
             "hard", "short", "late", "high", "fast", "different", "major", "minor"}


def to_adverb(adjective):
    """Turn an adjective into its adverb: careful -> carefully. None if no good form."""
    adj = adjective.lower()
    if adj in ADVERB_EXCEPTIONS:
        return ADVERB_EXCEPTIONS[adj]
    if adj in NO_ADVERB or adj.endswith("ly"):      # "timely", "friendly" have no -ly form
        return None
    if adj.endswith("ic"):
        return adj + "ally"                          # dramatic -> dramatically
    if adj.endswith("le") and len(adj) > 3:
        return adj[:-1] + "y"                        # simple -> simply
    if adj.endswith("ue"):
        return adj[:-1] + "ly"                       # true -> truly
    if adj.endswith("ll"):
        return adj + "y"                             # full -> fully
    if adj.endswith("y") and len(adj) > 2 and adj[-2] not in "aeiou":
        return adj[:-1] + "ily"                      # happy -> happily
    return adj + "ly"                                # careful -> carefully


def _way_suggestion(span):
    """Build "very carefully" from "in a very careful way"."""
    adj = [t for t in span if t.pos_ == "ADJ"][-1]
    adverb = to_adverb(adj.text)
    if adverb is None:
        return None
    modifiers = [t.text.lower() for t in span if t.pos_ == "ADV"]
    return " ".join(modifiers + [adverb])


# Build the matchers once.
_phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
_phrase_matcher.add("FIXED", list(nlp.tokenizer.pipe(FIXED_PHRASES.keys())))

_suggestions = {}   # flexible pattern name -> suggestion
_matcher = Matcher(nlp.vocab)
for name, suggestion, pattern in FLEXIBLE_PATTERNS:
    _matcher.add(name, [pattern])
    _suggestions[name] = suggestion
_matcher.add(WAY_PATTERN_NAME, [WAY_PATTERN])


def find_complex_phrases(sentence):
    """
    Find wordy or complex phrases that could be replaced with simpler ones.

    Returns a list of dictionaries, one per phrase found, for example:
        {"text": "To what extent", "pattern": "to what extent",
         "suggestion": "how much", "source": "manual", "start_index": 0, "end_index": 3,
         "start_char": 0, "end_char": 14, "first_tag": "TO",
         "sentence_start": True}

    Fields:
        text           - the phrase exactly as it appears in the sentence
        pattern        - which rule matched (the phrase from a CSV file, or a
                         FLEXIBLE_PATTERNS name)
        suggestion     - a simpler replacement (base form; may need inflecting)
        source         - where the phrase came from: "manual" or "simpleppdb"
        start_index    - index of the first word (same numbering as pos_tag)
        end_index      - index just AFTER the last word (like Python slicing)
        start_char     - character position where the phrase begins
        end_char       - character position where the phrase ends
        first_tag      - detailed POS tag of the first word (e.g. VBD = past tense),
                         useful for inflecting verb suggestions ("made a decision" -> "decided")
        sentence_start - True if the phrase starts a sentence (capitalize the replacement)

    If two matches overlap, a manual phrase always beats a SimplePPDB phrase;
    between phrases from the same source, the longer one wins.
    """
    doc = nlp(sentence)

    # Collect every match, with its suggestion and source
    matches = []
    for match_id, start, end in list(_phrase_matcher(doc)) + list(_matcher(doc)):
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        if label == "FIXED":
            pattern = " ".join(t.lower_ for t in span)
            suggestion, source = FIXED_PHRASES[pattern]
        else:
            if label == WAY_PATTERN_NAME:
                suggestion = _way_suggestion(span)
                if suggestion is None:
                    continue                 # e.g. "in a timely way": no good adverb
            else:
                suggestion = _suggestions[label]
            pattern, source = label, "manual"
        matches.append((span, pattern, suggestion, source))

    # Resolve overlaps: manual phrases always win over SimplePPDB phrases.
    # Within each source, the longest match wins.
    info = {}
    taken = set()          # word positions already covered by a chosen phrase
    chosen = []
    for source in ("manual", "simpleppdb"):
        spans = []
        for span, pattern, suggestion, src_name in matches:
            if src_name == source:
                info[(span.start, span.end)] = (pattern, suggestion, src_name)
                spans.append(span)
        for span in filter_spans(spans):
            if not taken.intersection(range(span.start, span.end)):
                chosen.append(span)
                taken.update(range(span.start, span.end))

    results = []
    for span in sorted(chosen, key=lambda s: s.start):
        pattern, suggestion, source = info[(span.start, span.end)]
        results.append({
            "text": span.text,
            "pattern": pattern,
            "suggestion": suggestion,
            "source": source,
            "start_index": span.start,
            "end_index": span.end,
            "start_char": span.start_char,
            "end_char": span.end_char,
            "first_tag": span[0].tag_,
            "sentence_start": span[0].is_sent_start,
        })
    return results



def print_phrases(sentence):
    """Print any complex phrases found (handy for testing)."""
    print(f"\nSentence: {sentence}")
    phrases = find_complex_phrases(sentence)
    if not phrases:
        print("  No complex phrases found.")
    for p in phrases:
        print(f'  "{p["text"]}" (words {p["start_index"]}-{p["end_index"] - 1})'
              f' -> suggest "{p["suggestion"]}" [{p["source"]}]')


if __name__ == "__main__":
    print_phrases("To what extent is climate change caused by human activity?")
    print_phrases("The board made a decision in order to reduce costs prior to the merger.")

    while True:
        sentence = input("\nEnter a sentence (or press Enter to quit): ").strip()
        if not sentence:
            break
        print_phrases(sentence)