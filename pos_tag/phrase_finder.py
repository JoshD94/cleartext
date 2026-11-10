"""
Complex phrase finder for the sentence simplification project.

Finds wordy or complex phrases (e.g. "to what extent", "in order to",
"made a decision") and suggests simpler replacements.

Setup: same as pos_tag.py (spaCy + en_core_web_sm). This file must be
in the same folder as pos_tag.py, because it reuses the loaded model.

Phrases come from these sources, in priority order:
    manual      manual_phrases.csv + rules (always reliable)
    verified    verified_phrases.csv (reviewed SimplePPDB phrases, if any)
    model       the trained CRF detector (phrase_model.py) with replacements
                from the learned table or BERT (phrase_replacer.py)
    simpleppdb  simpleppdb_phrases.csv (off by default: it tested at 0% precision)

To choose sources, set the PHRASE_LISTS environment variable, e.g.
    PHRASE_LISTS=manual,model python test_phrase_finder.py
(evaluate.py has a --lists option that does this for you.)

Usage from your teammate's code:
    from phrase_finder import find_complex_phrases
    phrases = find_complex_phrases("To what extent is this true?")
"""

import csv
import os
from pathlib import Path

from spacy.matcher import Matcher, PhraseMatcher
from spacy.util import filter_spans

from pos_tag import nlp  # reuse the model that pos_tag already loaded

# ---------------------------------------------------------------------------
# Phrase lists (loaded from CSV files in the same folder as this script)
#
#   manual_phrases.csv     - hand-written phrases (edit this to add your own)
#   verified_phrases.csv   - SimplePPDB phrases you reviewed and marked good,
#                            created by review_simpleppdb.py (optional)
#   simpleppdb_phrases.csv - phrases from the SimplePPDB++ dataset, created by
#                            running build_phrase_list.py (optional)
#
# Priority when phrases overlap: manual > verified > simpleppdb.
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
SOURCES = ("manual", "verified", "model", "simpleppdb")   # in priority order
DEFAULT_SOURCES = "manual,verified,model"
ACTIVE = [s.strip() for s in os.environ.get("PHRASE_LISTS", DEFAULT_SOURCES).split(",")
          if s.strip()]
PHRASE_FILES = [(s, HERE / f"{s}_phrases.csv") for s in SOURCES
                if s in ACTIVE and s != "model"]


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


_model_status = None      # None = not tried yet, True = ready, False = unavailable

# Confidence the CRF needs before a phrase is replaced. train_crf.py picks 0.1
# to score best on ASSET, but on real articles (eval_dev.csv) that flagged 28
# phrases with only 4 right; 0.3 flagged 10 with 3 right. Replacing text calls
# for fewer, more reliable phrases, so the pipeline uses 0.3.
MODEL_THRESHOLD = 0.3
_CLITICS = {"'d", "'s", "n't", "'re", "'ve", "'ll", "'m"}


def _plausible(span):
    """Skip spans that can't be a replaceable phrase: names, numbers, punctuation, "'d been"."""
    if span[0].lower_ in _CLITICS:
        return False
    return not any(t.is_punct or t.pos_ == "PROPN" or t.like_num for t in span)


def _model_matches(doc):
    """Complex phrases found by the trained CRF (empty if it isn't trained yet)."""
    global _model_status
    if _model_status is False:
        return []
    try:
        from phrase_model import detect_doc
        spans = [doc[a:b] for a, b in detect_doc(doc, MODEL_THRESHOLD)]
        spans = [s for s in spans if _plausible(s)]
        _model_status = True
        return spans
    except (FileNotFoundError, ImportError) as e:
        print(f"[phrase_finder] model source skipped: {e}")
        _model_status = False
        return []


def _replace(doc, start, end):
    """Replacement for a model-detected phrase: learned table, then BERT."""
    from phrase_replacer import suggest_for_doc
    return suggest_for_doc(doc, start, end)


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
        source         - where the phrase came from: "manual", "verified",
                         "model" or "simpleppdb"
        start_index    - index of the first word (same numbering as pos_tag)
        end_index      - index just AFTER the last word (like Python slicing)
        start_char     - character position where the phrase begins
        end_char       - character position where the phrase ends
        first_tag      - detailed POS tag of the first word (e.g. VBD = past tense),
                         useful for inflecting verb suggestions ("made a decision" -> "decided")
        sentence_start - True if the phrase starts a sentence (capitalize the replacement)

    If two matches overlap, the higher-priority source wins
    (manual > verified > model > simpleppdb); within a source, the longer one wins.
    For "model" phrases, "pattern" says how the replacement was found:
    "model (learned)" or "model (bert)".
    """
    doc = nlp(sentence)

    # Collect every match, with its suggestion and source
    matches = []
    flexible = list(_matcher(doc)) if "manual" in ACTIVE else []
    for match_id, start, end in list(_phrase_matcher(doc)) + flexible:
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

    # Resolve overlaps in priority order (manual > verified > model > simpleppdb).
    # Within each source, the longest match wins.
    model_matches = _model_matches(doc) if "model" in ACTIVE else []
    taken = set()          # word positions already covered by a chosen phrase
    chosen = []            # (span, pattern, suggestion, source)
    for source in SOURCES:
        if source == "model":
            # model phrases are only looked up where nothing else matched
            for span in model_matches:
                if not taken.intersection(range(span.start, span.end)):
                    suggestion, method = _replace(doc, span.start, span.end)
                    if suggestion:
                        chosen.append((span, f"model ({method})", suggestion, "model"))
                        taken.update(range(span.start, span.end))
            continue
        by_span = {}
        for span, pattern, suggestion, src_name in matches:
            if src_name == source:
                by_span[(span.start, span.end)] = (pattern, suggestion)
        for span in filter_spans([doc[a:b] for a, b in by_span]):
            if not taken.intersection(range(span.start, span.end)):
                pattern, suggestion = by_span[(span.start, span.end)]
                chosen.append((span, pattern, suggestion, source))
                taken.update(range(span.start, span.end))

    results = []
    for span, pattern, suggestion, source in sorted(chosen, key=lambda c: c[0].start):
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
