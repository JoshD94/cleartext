"""
Rule-based passive -> active voice converter.

Uses spaCy's dependency parse to find passive clauses and lemminflect to
re-conjugate the verb.

    "The cake was eaten by John."      -> "John ate the cake."
    "The house has been sold by them." -> "They have sold the house."
    "Mistakes were made."              -> unchanged (no agent), or
                                          "Someone made mistakes." with
                                          placeholder="someone"

Setup:
    pip install spacy lemminflect
    python -m spacy download en_core_web_sm
"""

from dataclasses import dataclass, field

import spacy
from lemminflect import getInflection

nlp = spacy.load("en_core_web_sm")

# Pronoun case swaps: the old subject becomes an object, and the old
# agent ("by him") becomes a subject.
SUBJ_TO_OBJ = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them"}
OBJ_TO_SUBJ = {v: k for k, v in SUBJ_TO_OBJ.items()}

MODALS = {"will", "would", "can", "could", "may", "might", "shall", "should", "must"}
REL_PRONOUNS = {"that", "which", "who", "whom"}
NO_SPACE_BEFORE = {",", ".", ";", ":", "?", "!", "'s", "n't", "'", ")", "%"}


@dataclass
class Result:
    original: str
    output: str
    changes: list = field(default_factory=list)  # (before_clause, after_clause)
    skipped: list = field(default_factory=list)  # (clause, reason)

    @property
    def changed(self):
        return self.output != self.original


# ---------------------------------------------------------------- helpers

def inflect(lemma, tag):
    forms = getInflection(lemma, tag=tag)
    return forms[0] if forms else lemma


def person_number(head):
    """Rough agreement class of a subject: '1sg', '3sg' or 'pl'."""
    w = head.lower_
    if w == "i":
        return "1sg"
    if w in ("you", "we", "they", "someone_pl"):
        return "pl"
    if any(c.dep_ == "conj" for c in head.children):  # "John and Mary"
        return "pl"
    if head.tag_ in ("NNS", "NNPS"):
        return "pl"
    return "3sg"


def agree(lemma, tense, pn):
    """Conjugate be/have/do or a main verb for tense ('pres'/'past') and subject."""
    if lemma == "be":
        if tense == "past":
            return "were" if pn == "pl" else "was"
        return {"1sg": "am", "3sg": "is"}.get(pn, "are")
    if tense == "past":
        return inflect(lemma, "VBD")
    return inflect(lemma, "VBZ") if pn == "3sg" else inflect(lemma, "VBP")


def tense_of(tag):
    return "past" if tag == "VBD" else "pres"


def phrase(tokens, case_map, lower_first=False):
    words = []
    for i, t in enumerate(tokens):
        w = t.text
        if t.tag_ == "PRP" and w.lower() in case_map:
            w = case_map[w.lower()]
            if w == "i":
                w = "I"
        elif i == 0 and lower_first and t.pos_ != "PROPN" and w != "I":
            w = w.lower()
        words.append(w)
    return words


def detokenize(words):
    out = ""
    for w in words:
        if not w:
            continue
        # Moving a clause can leave its closing comma before ".", "?" etc.
        if out.endswith(",") and w in {".", "?", "!", ",", ";", ":"}:
            out = out[:-1]
        if out and w not in NO_SPACE_BEFORE and not out.endswith(("(", "“", '"')):
            out += " "
        out += w
    return out


# ------------------------------------------------------------- core logic

def find_passives(doc):
    """
    Yield (verb, subj, auxpass) for every passive clause in the doc,
    main clauses first. Converting the outer clause first moves any inner
    clause as one block, which keeps the next parse cleaner.
    """
    found = []
    for tok in doc:
        auxpass = [c for c in tok.children if c.dep_ == "auxpass"]
        subj = [c for c in tok.children if c.dep_ == "nsubjpass"]
        if auxpass and subj:
            depth = sum(1 for _ in tok.ancestors)
            found.append((depth, tok.i, tok, subj[0], auxpass[0]))
    for _, _, tok, subj, auxpass in sorted(found, key=lambda f: f[:2]):
        yield tok, subj, auxpass


def build_verb_group(verb, auxpass, auxes, neg, inner_adverbs, pn):
    """
    Map the passive verb group onto an active one.

    The passive "be" carries the form the active main verb needs:
        was eaten        -> ate          (VBD)
        is eaten         -> eats         (VBZ/VBP, re-agreed)
        is being eaten   -> is eating    (VBG)
        has been eaten   -> has eaten    (VBN)
        will be eaten    -> will eat     (VB)
    """
    lemma = verb.lemma_
    form = auxpass.tag_
    other = [a for a in auxes if a.i != auxpass.i]  # has, will, was (in "was being")
    words = []

    for i, a in enumerate(other):
        if i == 0 and a.tag_ in ("VBZ", "VBP", "VBD") and a.lower_ not in MODALS:
            words.append(agree(a.lemma_, tense_of(a.tag_), pn))  # re-agree finite aux
        else:
            words.append(a.text.lower() if i > 0 or a.lower_ in MODALS else a.text)

    if not other and form in ("VBD", "VBZ", "VBP"):
        # The passive "be" was the finite verb.
        if neg:  # "was not eaten" -> "did not eat"
            words.append(agree("do", tense_of(form), pn))
            words.append(neg.text)
            words.extend(a.text for a in inner_adverbs)
            words.append(lemma)
            return words
        words.extend(a.text for a in inner_adverbs)
        words.append(agree(lemma, tense_of(form), pn))
        return words

    if neg:
        words.append(neg.text)
    words.extend(a.text for a in inner_adverbs)
    words.append(inflect(lemma, {"VBG": "VBG", "VBN": "VBN"}.get(form, "VB")))
    return words


def convert_one(doc, placeholder=None):
    """
    Convert the first convertible passive clause.
    Returns (new_text, before_clause, after_clause, skipped) or None.
    """
    skipped = []
    for verb, subj, auxpass in find_passives(doc):
        clause = doc[verb.left_edge.i: verb.right_edge.i + 1].text
        auxes = sorted(
            (c for c in verb.children if c.dep_ in ("aux", "auxpass") and c.i < verb.i),
            key=lambda t: t.i,
        )
        # Questions / inversion ("Was the cake eaten by John?") are left alone.
        if any(a.i < subj.i for a in auxes):
            skipped.append((clause, "question or inverted word order"))
            continue
        if any(a.lower_ in ("to", "get", "got", "gets") for a in auxes) or auxpass.lemma_ != "be":
            skipped.append((clause, "infinitive or get-passive"))
            continue

        agent = next((c for c in verb.children if c.dep_ == "agent"), None)
        agent_obj = None
        if agent is not None:
            agent_obj = next((c for c in agent.children if c.dep_ == "pobj"), None)
        if agent_obj is None and not placeholder:
            skipped.append((clause, "no agent (no 'by ...' phrase)"))
            continue

        neg = next((c for c in verb.children if c.dep_ == "neg"), None)
        inner_adverbs = [
            c for c in verb.children
            if c.dep_ == "advmod" and auxes and auxes[0].i < c.i < verb.i
        ]
        particle = next((c for c in verb.children if c.dep_ == "prt"), None)

        # New subject = old agent (or placeholder).
        sentence_start = subj.left_edge.i == subj.sent.start
        if agent_obj is not None:
            new_subj_toks = list(agent_obj.subtree)
            new_subj = phrase(new_subj_toks, OBJ_TO_SUBJ)
            pn = person_number(agent_obj)
            if agent_obj.lower_ in OBJ_TO_SUBJ:  # "by me" -> "I", "by them" -> "they"
                pn = {"i": "1sg", "he": "3sg", "she": "3sg"}.get(
                    OBJ_TO_SUBJ[agent_obj.lower_], "pl")
        else:
            new_subj = [placeholder]
            pn = "3sg"
        if sentence_start and new_subj:
            new_subj[0] = new_subj[0][0].upper() + new_subj[0][1:]

        # New object = old subject.
        subj_toks = list(subj.subtree)
        relative = len(subj_toks) == 1 and subj.lower_ in REL_PRONOUNS
        # "It was decided by X that ..." -> "X decided that ...": the "it"
        # is a placeholder for the that-clause, so it is dropped.
        expletive = subj.lower_ == "it" and any(
            c.dep_ in ("ccomp", "csubjpass") and c.i > verb.i for c in verb.children)
        if relative or expletive:
            new_obj = []
        else:
            new_obj = phrase(subj_toks, SUBJ_TO_OBJ, lower_first=sentence_start)

        verb_group = build_verb_group(verb, auxpass, auxes, neg, inner_adverbs, pn)
        if particle is not None:
            verb_group.append(particle.text)

        # Tokens that get moved or dropped.
        removed = {t.i for t in subj_toks} | {a.i for a in auxes} | {a.i for a in inner_adverbs}
        if neg is not None:
            removed.add(neg.i)
        if particle is not None:
            removed.add(particle.i)
        if agent is not None:
            removed |= {t.i for t in agent.subtree}
        removed.discard(verb.i)

        words, clause_words = [], []
        for t in doc:
            if t.i == subj.left_edge.i:
                block = ([subj.text] if relative else []) + new_subj
                words += block
                clause_words += block
            if t.i == verb.i:
                block = verb_group + new_obj
                words += block
                clause_words += block
                continue
            if t.i in removed:
                continue
            words.append(t.text)
            if verb.left_edge.i <= t.i <= verb.right_edge.i:
                clause_words.append(t.text)

        return detokenize(words), clause, detokenize(clause_words), skipped

    return None, None, None, skipped


def to_active(text, placeholder=None, max_passes=5):
    """
    Convert passive clauses in `text` to active voice.

    placeholder: if set (e.g. "someone"), agentless passives are converted
                 using it as the subject. Otherwise they are left unchanged.
    """
    result = Result(original=text, output=text)
    current = text
    for _ in range(max_passes):
        doc = nlp(current)
        new_text, before, after, skipped = convert_one(doc, placeholder)
        if new_text is None:
            result.skipped = skipped
            break
        result.changes.append((before, after))
        current = new_text
    result.output = current
    return result


if __name__ == "__main__":
    import sys

    sentences = sys.argv[1:] or ["The cake was eaten by John."]
    for s in sentences:
        r = to_active(s)
        print(f"{s}\n  -> {r.output}")
        for clause, reason in r.skipped:
            print(f"     (skipped: {reason})")
