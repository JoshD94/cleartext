"""
Rule-based passive -> active voice converter.

Uses spaCy's dependency parse to find passive clauses and lemminflect to
re-conjugate the verb.

    from passive_to_active import PassiveToActive

    p2a = PassiveToActive()
    p2a.transform("The cake was eaten by John.")       # "John ate the cake."
    p2a.transform("The house has been sold by them.")  # "They have sold the house."
    p2a.transform("Mistakes were made.")               # "Someone made mistakes."
    p2a.transform("She was born in 1990.")             # unchanged (a state)

The PassiveToActive class at the bottom of this file is the public API.
Everything above it is internal.

Setup:
    pip install spacy lemminflect
    python -m spacy download en_core_web_sm
"""

from dataclasses import dataclass, field

import spacy
from lemminflect import getAllLemmas, getInflection

nlp = spacy.load("en_core_web_sm")

# Pronoun case swaps: the old subject becomes an object, and the old
# agent ("by him") becomes a subject.
SUBJ_TO_OBJ = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them"}
OBJ_TO_SUBJ = {v: k for k, v in SUBJ_TO_OBJ.items()}

MODALS = {"will", "would", "can", "could", "may", "might", "shall", "should", "must"}
REL_PRONOUNS = {"that", "which", "who", "whom"}
# Passives that describe a state rather than an action. Without a "by"
# phrase these have no natural active form ("She was born in 1990" is not
# "Someone bore her in 1990"), so they are never given a placeholder subject.
# None of these block conversion when there IS a "by" phrase.
STATIVE_LEMMAS = {
    "bear", "marry", "divorce", "engage", "locate", "situate", "base",
    "suppose", "involve", "relate", "dress", "name", "call", "compose",
    "interest", "concern", "tire", "bore", "scare", "frighten", "worry",
    "please", "satisfy", "disappoint", "excite", "amaze", "shock", "embarrass",
    "confuse", "puzzle", "delight", "thrill", "upset", "annoy", "depress",
}
# Lemmas that are stative only with a certain preposition:
# "made of wood", "known for his music", "covered in dust".
STATIVE_WITH_PREP = {
    "make": {"of", "from"},
    "know": {"for", "as"},
    "cover": {"in", "with"},
    "fill": {"with"},
    "use": {"to"},
    "equip": {"with"},
}

NO_SPACE_BEFORE ={",", ".", ";", ":", "?", "!", "'s", "n't", "'", ")", "%"}


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


def is_real_proper_noun(tok):
    """
    Capitalized words at the start of a sentence are often mis-tagged as
    proper nouns ("Rice is eaten..."). Treat a sentence-initial "proper noun"
    as a common noun if it's in the English dictionary and isn't part of a
    longer name. Trade-off: names that are also words ("Bill", "Mark") get
    lowercased when they start a sentence alone.
    """
    if tok.pos_ != "PROPN":
        return False
    if tok.i != tok.sent.start:
        return True
    nxt = tok.doc[tok.i + 1] if tok.i + 1 < len(tok.doc) else None
    if nxt is not None and nxt.text[:1].isupper():  # "Apple Music", "New York"
        return True
    return not getAllLemmas(tok.text.lower())


def phrase(tokens, case_map, lower_first=False):
    words = []
    for i, t in enumerate(tokens):
        w = t.text
        if t.tag_ == "PRP" and w.lower() in case_map:
            w = case_map[w.lower()]
            if w == "i":
                w = "I"
        elif i == 0 and lower_first and w != "I" and not is_real_proper_noun(t):
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


def build_verb_group(verb, auxpass, auxes, neg, inner_adverbs, pn, question=False):
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
        if neg or question:  # "was not eaten" -> "did not eat"; questions need "do"
            words.append(agree("do", tense_of(form), pn))
            if neg:
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


def agentless_skip_reason(verb, auxpass, auxes, placeholder):
    """Why an agentless passive should be left alone, or None to convert it."""
    if not placeholder:
        return "no agent (no 'by ...' phrase)"
    if verb.lemma_ in STATIVE_LEMMAS:
        return "describes a state, not an action"
    preps = {c.lower_ for c in verb.children if c.dep_ in ("prep", "aux")}
    if preps & STATIVE_WITH_PREP.get(verb.lemma_, set()):
        return "describes a state, not an action"
    if any(c.dep_ in ("xcomp", "oprd") for c in verb.children):
        # "is said to be rich", "are not allowed to smoke", "is located downtown"
        return "passive with a following verb or complement"
    return None


def choose_placeholder(auxpass, auxes, placeholder):
    """
    placeholder="auto": "people" for general facts in the simple present
    ("English is spoken in Canada" -> "People speak English in Canada"),
    "someone" for everything else ("The thief was caught" -> "Someone caught
    the thief"). Any other value is used as given.
    """
    if placeholder != "auto":
        return placeholder
    simple_present = auxpass.tag_ in ("VBZ", "VBP") and len(auxes) == 1
    return "people" if simple_present else "someone"


def convert_one(doc, placeholder=None, question_prefix=None):
    """
    Convert the first convertible passive clause.
    Returns (new_text, before_clause, after_clause, skipped) or None.

    question_prefix: when set (a list of words, possibly empty, such as
    ["Why"]), `doc` is a statement made from a question, and the result is
    put back into question order.
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
        if agent_obj is None:
            reason = agentless_skip_reason(verb, auxpass, auxes, placeholder)
            if reason:
                skipped.append((clause, reason))
                continue

        neg = next((c for c in verb.children if c.dep_ == "neg"), None)
        inner_adverbs = [
            c for c in verb.children
            if c.dep_ == "advmod" and auxes and auxes[0].i < c.i < verb.i
        ]
        particle = next((c for c in verb.children if c.dep_ == "prt"), None)

        # "What was eaten by John?" -> "What did John eat?"
        wh_subject = (
            question_prefix is None
            and doc[-1].text == "?"
            and subj.left_edge.i == 0
            and subj.left_edge.tag_ in ("WDT", "WP", "WP$")
        )
        question = question_prefix is not None or wh_subject

        # New subject = old agent (or placeholder).
        sentence_start = subj.left_edge.i == subj.sent.start and not question
        if agent_obj is not None:
            new_subj_toks = list(agent_obj.subtree)
            new_subj = phrase(new_subj_toks, OBJ_TO_SUBJ)
            pn = person_number(agent_obj)
            if agent_obj.lower_ in OBJ_TO_SUBJ:  # "by me" -> "I", "by them" -> "they"
                pn = {"i": "1sg", "he": "3sg", "she": "3sg"}.get(
                    OBJ_TO_SUBJ[agent_obj.lower_], "pl")
        else:
            word = choose_placeholder(auxpass, auxes, placeholder)
            new_subj = [word]
            pn = "pl" if word in ("people", "they", "we", "you") else "3sg"
        if sentence_start and new_subj:
            new_subj[0] = new_subj[0][0].upper() + new_subj[0][1:]

        # New object = old subject.
        subj_toks = list(subj.subtree)
        relative = len(subj_toks) == 1 and subj.lower_ in REL_PRONOUNS and not wh_subject
        # "It was decided by X that ..." -> "X decided that ...": the "it"
        # is a placeholder for the that-clause, so it is dropped.
        expletive = subj.lower_ == "it" and any(
            c.dep_ in ("ccomp", "csubjpass") and c.i > verb.i for c in verb.children)
        if relative or expletive:
            new_obj = []
        else:
            new_obj = phrase(subj_toks, SUBJ_TO_OBJ,
                             lower_first=sentence_start or question_prefix is not None)

        verb_group = build_verb_group(verb, auxpass, auxes, neg, inner_adverbs, pn,
                                      question=question)
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

        if question:
            # Front the first helper verb: "did John eat", "has she sold".
            front, tail = verb_group[0], verb_group[1:]
            if tail and tail[0] == "n't":  # "didn't John eat"
                front, tail = front + "n't", tail[1:]
            after = [t.text for t in doc
                     if t.i > verb.i and t.i not in removed and not t.is_punct]
            if wh_subject:
                words = new_obj + [front] + new_subj + tail + after + ["?"]
            else:
                words = list(question_prefix) + [front] + new_subj + tail + new_obj + after + ["?"]
            words[0] = words[0][0].upper() + words[0][1:]
            text = detokenize(words)
            return text, clause, text, skipped

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


FRONTABLE = {"am", "is", "are", "was", "were", "has", "have", "had"} | MODALS
VERB_PART = {"be", "been", "being", "not", "n't"}
WH_AGENT_PLACEHOLDER = "someone"


def question_to_statement(text):
    """
    Undo the inversion in a passive question so it can be parsed reliably.

        "Was the cake eaten by John?"   -> ([], None, "the cake was eaten by John.")
        "Why was the cake eaten by X?"  -> (["Why"], None, "the cake was eaten by X.")
        "Who was the cake eaten by?"    -> ([], ["Who"], "the cake was eaten by someone.")
        "By whom was the cake eaten?"   -> ([], ["Who"], "the cake was eaten by someone.")

    Returns (prefix, wh_agent, statement), or None if `text` isn't an
    inverted question.
    """
    toks = list(nlp(text))
    if len(toks) < 4 or toks[-1].text != "?":
        return None
    aux_i = next((i for i, t in enumerate(toks) if t.lower_ in FRONTABLE), None)
    if aux_i is None or aux_i > 3:
        return None
    prefix = [t.text for t in toks[:aux_i]]

    j = aux_i + 1
    neg = []
    if toks[j].lower_ == "n't":
        neg, j = [toks[j].text], j + 1

    # The subject runs until the next verb part, adverb or participle.
    k = j
    while k < len(toks) - 1 and not (
        toks[k].lower_ in VERB_PART or toks[k].tag_ in ("VBN", "VBD", "RB", "MD")
    ):
        k += 1
    if k == j:
        return None
    subj = [t.text for t in toks[j:k]]
    rest = [t.text for t in toks[k:-1]]
    if subj[0] != "I" and toks[j].pos_ != "PROPN":
        subj[0] = subj[0].lower()

    # A question word asking about the agent: "By whom ...?" / "Who ... by?"
    wh_agent = None
    if len(prefix) >= 2 and prefix[0].lower() == "by":
        wh_agent, prefix = prefix[1:], []
    elif prefix and rest and rest[-1].lower() == "by":
        wh_agent, prefix, rest = prefix, [], rest[:-1]
    if wh_agent:
        wh_agent = ["Who" if w.lower() == "whom" else w for w in wh_agent]
        rest += ["by", WH_AGENT_PLACEHOLDER]

    statement = detokenize(subj + [toks[aux_i].lower_] + neg + rest + ["."])
    return prefix, wh_agent, statement


def question_to_active(text, placeholder=None):
    """Convert an inverted passive question. Returns a Result or None."""
    parsed = question_to_statement(text)
    if parsed is None:
        return None
    prefix, wh_agent, statement = parsed
    doc = nlp(statement)
    result = Result(original=text, output=text)

    if wh_agent:
        # "the cake was eaten by someone." -> "Someone ate the cake." -> "Who ate the cake?"
        new_text, before, _, skipped = convert_one(doc)
        if new_text is None or not new_text.lower().startswith(WH_AGENT_PLACEHOLDER):
            result.skipped = skipped or [(text, "could not convert question")]
            return result
        wh = " ".join(wh_agent)
        new_text = wh[0].upper() + wh[1:] + new_text[len(WH_AGENT_PLACEHOLDER):]
        new_text = new_text.rstrip(".") + "?"
    else:
        new_text, before, _, skipped = convert_one(doc, placeholder, question_prefix=prefix)
        if new_text is None:
            result.skipped = skipped
            return result

    result.output = new_text
    result.changes.append((text, new_text))
    return result


def to_active(text, placeholder="auto", max_passes=5):
    """
    Convert passive clauses in one sentence to active voice.

    placeholder: what to do with passives that have no "by" phrase.
        "auto" (default) - use "people" for general present-tense facts and
                           "someone" otherwise; skip passives that describe
                           a state ("She was born in 1990").
        None             - leave them unchanged.
        any word       - use that word as the subject (state passives are
                         still skipped).
    """
    question = question_to_active(text, placeholder)
    if question is not None:
        return question

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


# ------------------------------------------------------------ public API

class PassiveToActive:
    """
    Turns passive sentences into active ones. This is the piece to plug
    into the bigger simplification tool.

        from passive_to_active import PassiveToActive

        p2a = PassiveToActive()
        p2a.transform("The cake was eaten by John.")    # "John ate the cake."
        p2a.transform("John ate the cake.")             # unchanged
        p2a.transform_text("The thief was caught. He was sent to jail by the judge.")
        # "Someone caught the thief. The judge sent him to jail."

    Sentences that aren't passive, or can't be converted safely, come back
    exactly as they went in, so it's safe to run on every sentence.

    placeholder: subject for passives with no "by" phrase.
        "auto" (default) - "people" for general facts, "someone" otherwise
        None             - leave those sentences unchanged
        any word         - always use that word
    """

    name = "passive_to_active"

    def __init__(self, placeholder="auto"):
        self.placeholder = placeholder

    def transform(self, sentence):
        """One sentence in, one sentence out."""
        if not sentence or not sentence.strip():
            return sentence
        return to_active(sentence, placeholder=self.placeholder).output

    def transform_text(self, text):
        """A paragraph or longer text: splits it into sentences and transforms each."""
        out = []
        for sent in nlp(text).sents:
            trailing = sent.text_with_ws[len(sent.text):]
            out.append(self.transform(sent.text) + trailing)
        return "".join(out)

    def is_passive(self, sentence):
        """True if the sentence contains a passive clause (converted or not)."""
        if any(True for _ in find_passives(nlp(sentence))):
            return True
        parsed = question_to_statement(sentence)
        return bool(parsed) and any(True for _ in find_passives(nlp(parsed[2])))

    def explain(self, sentence):
        """
        Full details for debugging: a Result with .output, .changed,
        .changes (before/after clauses) and .skipped (clause, reason).
        """
        return to_active(sentence, placeholder=self.placeholder)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    placeholder = "auto"
    if args and args[0].startswith("--placeholder="):
        value = args.pop(0).split("=", 1)[1]
        placeholder = None if value.lower() == "none" else value
    p2a = PassiveToActive(placeholder=placeholder)
    sentences = args or ["The cake was eaten by John."]
    for s in sentences:
        r = p2a.explain(s)
        print(f"{s}\n  -> {r.output}")
        for clause, reason in r.skipped:
            print(f"     (skipped: {reason})")