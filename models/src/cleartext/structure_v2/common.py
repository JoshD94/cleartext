"""Shared helpers for the structure v2 operations.

Every operation rewrites one sentence at a time from a spaCy parse and
rebuilds the text from the original tokens and their own whitespace, so
hyphenated words, quotes and brackets come back byte for byte. Inserted
pieces get one space before them unless they follow an opening bracket or
quote.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

from lemminflect import getInflection

from ..features import nlp

OPENERS = ("(", "[", "“", '"', "‘", "'")
SUBJ_TO_OBJ = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them", "who": "whom"}
OBJ_TO_SUBJ = {"me": "I", "him": "he", "her": "she", "us": "we", "them": "they", "whom": "who"}
# Nouns that take plural agreement without a plural tag.
PLURAL_NOUNS = {"police", "people", "cattle", "personnel", "clergy", "authorities"}
HARD_CHECKS = ("numbers", "negation", "entities", "technical_phrases", "code_literals")


@dataclass
class Edit:
    """One accepted operation on one sentence."""
    op: str
    rule: str
    before: str
    after: str
    features: dict = field(default_factory=dict)

    def as_dict(self):
        return {"stage": "structure_v2", "op": self.op, "rule": self.rule,
                "before": self.before, "after": self.after, "features": self.features}


class Pieces:
    """Builds a sentence from original tokens (with their own spacing) and
    inserted strings (one space before, none after an opener)."""

    def __init__(self):
        self.parts = []
        self.last = None  # index of the last original token emitted

    def token(self, tok, text=None):
        text = tok.text if text is None else text
        if self.parts:
            prev = tok.doc[tok.i - 1] if tok.i > 0 else None
            gap = prev.whitespace_ if prev is not None else ""
            if self.last != tok.i - 1:
                # Something was removed or inserted in between: keep the
                # token's own attachment (no space before "," or ")").
                gap = gap if gap == "" else " "
                if self.parts[-1].endswith(OPENERS):
                    gap = ""
            self.parts.append(gap)
        self.parts.append(text)
        self.last = tok.i

    def insert(self, text):
        if not text:
            return
        if self.parts and not self.parts[-1].endswith(OPENERS) and not self.parts[-1].endswith(" "):
            self.parts.append(" ")
        self.parts.append(text)
        self.last = None

    def text(self):
        out = "".join(self.parts)
        out = re.sub(r" {2,}", " ", out)
        out = re.sub(r" +([,.;:?!])", r"\1", out)
        out = re.sub(r",\s*([.;:?!])", r"\1", out)
        return out.strip()


def span_text(tokens, mapper=None):
    """Original text of contiguous tokens, optionally mapping individual words."""
    tokens = list(tokens)
    out = []
    for k, t in enumerate(tokens):
        w = mapper(t) if mapper else t.text
        out.append(w)
        if k < len(tokens) - 1:
            out.append(t.whitespace_)
    return "".join(out)


FUNCTION_POS = ("DET", "PRON", "ADP", "CCONJ", "SCONJ", "AUX", "PART")


def keeps_capital(tok):
    """True if a sentence-initial token must stay capitalized when moved."""
    w = tok.text
    if w != "I" and tok.pos_ in FUNCTION_POS and tok.is_stop:
        return False
    if w == "I" or len(w) > 1 and (w.isupper() or any(c.isupper() for c in w[1:])):
        return True
    if tok.ent_type_ in ("PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"):
        return True
    if tok.pos_ == "PROPN":
        nxt = tok.doc[tok.i + 1] if tok.i + 1 < len(tok.doc) else None
        if nxt is not None and nxt.text[:1].isupper():
            return True
        from lemminflect import getAllLemmas
        return not getAllLemmas(w.lower())
    return False


def lower_first(text, tok):
    if not text or keeps_capital(tok):
        return text
    return text[0].lower() + text[1:]


def upper_first(text):
    return text[:1].upper() + text[1:] if text else text


def inflect(lemma, tag):
    forms = getInflection(lemma, tag=tag)
    return forms[0] if forms else None


def number_of(head):
    """'1sg', '3sg' or 'pl' agreement class of a noun phrase head."""
    w = head.lower_
    if w == "i":
        return "1sg"
    if w in ("you", "we", "they"):
        return "pl"
    conj = [c for c in head.children if c.dep_ == "conj"]
    if conj:
        ccs = [t.lower_ for t in head.subtree if t.dep_ == "cc"]
        if "and" in ccs or not ccs:
            return "pl"
    if head.tag_ in ("NNS", "NNPS") or w in PLURAL_NOUNS:
        return "pl"
    return "3sg"


def agree(lemma, tense, pn):
    """be/have/do or a main verb, for tense 'past'/'pres' and agreement class."""
    if lemma == "be":
        if tense == "past":
            return "were" if pn == "pl" else "was"
        return {"1sg": "am", "3sg": "is"}.get(pn, "are")
    if tense == "past":
        return inflect(lemma, "VBD")
    return inflect(lemma, "VBZ" if pn == "3sg" else "VBP")


def in_quotes(tok):
    """True if the token sits inside a double-quoted stretch of its sentence."""
    sent = tok.sent
    count = sum(1 for t in sent if t.i < tok.i and t.text in ('"', "“", "”"))
    opened = sum(1 for t in sent if t.i < tok.i and t.text == "“")
    closed = sum(1 for t in sent if t.i < tok.i and t.text == "”")
    return opened > closed or (count - opened - closed) % 2 == 1


def hard_checks(source, output):
    """The five F2 hard checks (numbers, negation, entities, technical
    phrases, code literals); returns the names that fail."""
    from ..protected_literals import preserved_literals
    from ..structure.common import structure_checks
    checks = structure_checks(source, output)["checks"]
    failed = [k for k in ("numbers", "negation", "entities", "technical_phrases") if not checks[k]]
    if not preserved_literals(source, output)["pass"]:
        failed.append("code_literals")
    return failed


@lru_cache(maxsize=20000)
def parse(text):
    """Cached spaCy parse of a sentence or passage (the pipeline's model)."""
    return nlp()(text)


def sentence_spans(text):
    return [(s.start_char, s.end_char) for s in parse(text).sents]
