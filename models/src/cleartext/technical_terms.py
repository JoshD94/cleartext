"""Term spans from known phrases, acronym definitions and noun compounds.

Known phrases and explicit acronym definitions are protected. Other noun phrases
are proposals for inspection, not certified technical terms or automatic bans.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import re

from .features import nlp
from .lexical import PROTECTED


@dataclass(frozen=True)
class TermSpan:
    start: int
    end: int
    text: str
    reasons: tuple
    protected: bool

    def record(self):
        return asdict(self)


def surface_pattern(text):
    words = re.split(r"[\s-]+", text.strip())
    return r"(?<!\w)" + r"[\s-]+".join(re.escape(word) for word in words) + r"(?!\w)"


@lru_cache(1)
def known_lemmas():
    return [
        tuple(token.lemma_.lower() for token in doc if token.is_alpha)
        for doc in nlp().pipe(PROTECTED)
    ]


def acronym_matches(short, tokens):
    if not short.isupper() or not 2 <= len(short) <= 10 or not short.isalpha():
        return False
    initials = "".join(
        token.text[0] for token in tokens if token.is_alpha and not token.is_stop
    )
    return initials.casefold() == short.casefold()


def detect_terms(doc):
    found = {}

    def add(start, end, reason, protected):
        key = start, end
        previous = found.get(key)
        reasons = set(previous.reasons if previous else ()) | {reason}
        found[key] = TermSpan(
            start,
            end,
            doc.text[start:end],
            tuple(sorted(reasons)),
            protected or bool(previous and previous.protected),
        )

    for phrase in PROTECTED:
        for match in re.finditer(surface_pattern(phrase), doc.text, re.I):
            add(match.start(), match.end(), "known_phrase", True)
    # Match plural forms without adding a new hand-written phrase inventory.
    alpha = [token for token in doc if token.is_alpha]
    for words in known_lemmas():
        if not words:
            continue
        for start in range(len(alpha) - len(words) + 1):
            tokens = alpha[start : start + len(words)]
            connected = all(
                re.fullmatch(r"[\s-]+", doc.text[left.idx + len(left) : right.idx])
                for left, right in zip(tokens, tokens[1:])
            )
            if connected and tuple(token.lemma_.lower() for token in tokens) == words:
                add(
                    tokens[0].idx,
                    tokens[-1].idx + len(tokens[-1]),
                    "known_phrase",
                    True,
                )
    for token in doc:
        if token.text != "(":
            continue
        close = next(
            (
                end
                for end in range(token.i + 1, min(len(doc), token.i + 12))
                if doc[end].text == ")"
            ),
            None,
        )
        if close is None:
            continue
        inside = list(doc[token.i + 1 : close])
        long_tokens, short = [], None
        if len(inside) == 1:
            short = inside[0]
            for size in range(2, min(10, token.i) + 1):
                candidate = list(doc[token.i - size : token.i])
                if all(t.is_alpha for t in candidate) and acronym_matches(
                    short.text, candidate
                ):
                    long_tokens = candidate
                    break
        elif token.i:
            short = doc[token.i - 1]
            if all(t.is_alpha for t in inside) and acronym_matches(short.text, inside):
                long_tokens = inside
        if not long_tokens:
            continue
        add(
            long_tokens[0].idx,
            long_tokens[-1].idx + len(long_tokens[-1]),
            "acronym_definition",
            True,
        )
        # Protect exact acronym occurrences only after validating the definition.
        for match in re.finditer(surface_pattern(short.text), doc.text):
            add(match.start(), match.end(), "defined_acronym", True)
    for head in doc:
        if head.pos_ not in {"NOUN", "PROPN"}:
            continue
        modifiers = sorted(
            [
                child
                for child in head.children
                if child.dep_ in {"compound", "amod"} and child.i < head.i
            ],
            key=lambda t: t.i,
        )
        if not modifiers:
            continue
        begin = modifiers[0].i
        tokens = doc[begin : head.i + 1]
        if 2 <= len(tokens) <= 6 and all(t.is_alpha for t in tokens):
            add(tokens[0].idx, head.idx + len(head), "noun_phrase_proposal", False)
    return sorted(found.values(), key=lambda span: (span.start, span.end))


def preserved_terms(original_doc, output, spans=None):
    spans = detect_terms(original_doc) if spans is None else spans
    counts = Counter(
        (
            re.sub(
                r"[\s-]+",
                " ",
                span.text
                if "defined_acronym" in span.reasons
                else span.text.casefold(),
            ),
            "defined_acronym" in span.reasons,
        )
        for span in spans
        if span.protected
    )
    missing = [
        term
        for (term, case_sensitive), count in counts.items()
        if len(
            list(
                re.finditer(
                    surface_pattern(term), output, 0 if case_sensitive else re.I
                )
            )
        )
        < count
    ]
    return {"pass": not missing, "missing": missing, "semantic_certified": False}
