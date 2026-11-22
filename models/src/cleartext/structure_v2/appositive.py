"""Long appositives and non-restrictive relative clauses become their own
sentence (structure v2, opt-in).

    "Joel Anderson, a member of the State Assembly, won the primary."
 -> "Joel Anderson won the primary. Joel Anderson is a member of the State Assembly."
    "The plan, which was approved in May, cuts costs."
 -> "The plan cuts costs. The plan was approved in May."

Only subject-side or sentence-final material set off by commas is moved,
and only when the anchor noun phrase can be repeated verbatim (an "a/an"
anchor is repeated with "the"). The relative pronoun must be the clause's
subject. Names, codes and short descriptions stay.
"""

from .common import parse, Edit, Pieces, agree, lower_first, number_of, span_text, upper_first

QUOTES = ('"', "“", "”")
MIN_APPOS = 4  # words in the appositive


def _anchor_text(anchor, stop):
    """The anchor noun phrase up to (not including) token index `stop`."""
    toks = [t for t in anchor.doc[anchor.left_edge.i: stop]]
    while toks and toks[-1].is_punct:
        toks.pop()
    if not toks:
        return None
    text = span_text(toks)
    first = toks[0]
    if first.lower_ in ("a", "an"):
        text = "the" + text[len(first.text):]
    elif first.i == first.sent.start:
        text = lower_first(text, first)
    return text


def _main_tense(sent):
    root = sent.root
    tags = [root.tag_] + [c.tag_ for c in root.children if c.dep_ in ("aux", "auxpass")]
    if any(t in ("VBD",) for t in tags):
        return "past"
    return "pres"


def _comma_set(span_start, span_end, sent):
    """True if the span is preceded by a comma and followed by a comma or
    the sentence-final punctuation."""
    doc = sent.doc
    before = doc[span_start - 1] if span_start - 1 >= sent.start else None
    after = doc[span_end + 1] if span_end + 1 < sent.end else None
    return (before is not None and before.text == ","
            and after is not None and (after.text == "," or after.i == sent.end - 1 and after.is_punct))


def _bad_span(toks):
    return any(t.text in ("(", ")", ";", ":", "—", "–", '"', "“", "”") or t.like_num and t.i == toks[0].i
               for t in toks)


PAST_ROLE = {"former", "late", "then", "previous", "onetime", "erstwhile"}
FINITE = ("VBD", "VBZ", "VBP", "MD")


def _risky(kind, tok, anchor, span, start, sent, final):
    """Round-2 skips, one per failure type seen in round 1."""
    from .passive import run_in_heading
    if run_in_heading(sent):
        return True  # "Esomeprazole The data described below ..."
    if kind == "relative" and span[0].lower_ == "which":
        # "which" pointing at a whole clause: the anchor has its own clause
        # ("the songs they wrote, which ..."), sits inside a gerund phrase
        # ("minimizing regulatory burden, which ..."), or "which in turn".
        own_tokens = [t for t in anchor.subtree if t.i < start - 1]
        if any(t.pos_ in ("VERB", "AUX") for t in own_tokens):
            return True
        if any(a.tag_ == "VBG" and a.dep_ != "ROOT" for a in anchor.ancestors):
            return True
        if len(span) > 2 and span[1].lower_ in ("in", "then", "thus", "therefore", "also"):
            return True
    if any(c.dep_ == "conj" for c in tok.children) or any(t.dep_ == "cc" and t.head.i == tok.i for t in span):
        return True  # a list, not an appositive ("an organization, an organization, and an individual")
    own = [t for t in anchor.subtree if t.i < start - 1]
    if any(t.dep_ == "prep" and t.head.i == anchor.i for t in own):
        return True  # "a member of the family, the most prominent ..." may describe either noun
    if anchor.dep_ == "pobj" and anchor.head.head.pos_ not in ("VERB", "AUX"):
        return True  # "a loss against Molla Mallory, her only amateur match"
    if any(a.dep_ == "advcl" for a in anchor.ancestors):
        return True  # the anchor sits inside an adverbial clause ("Although the eruption ... PST, ...")
    if kind == "appositive":
        for t in span:
            if t.tag_ in FINITE and not any(a.dep_ in ("relcl", "acl") and a.i in {x.i for x in span}
                                            for a in [t, *t.ancestors]):
                return True  # a finite verb: probably the main clause, misparsed
    if kind == "relative":
        verb_tags = [tok.tag_] + [c.tag_ for c in tok.children if c.dep_ in ("aux", "auxpass")]
        finite = next((t for t in verb_tags if t in ("VBZ", "VBP")), None)
        aux = [c.lower_ for c in tok.children if c.dep_ in ("aux", "auxpass")] + [tok.lower_]
        plural_verb = finite == "VBP" or "were" in aux
        singular_verb = finite == "VBZ" or "was" in aux
        pn = number_of(anchor)
        if plural_verb and pn != "pl" or singular_verb and pn == "pl":
            return True  # "a Hispanic audience, who were ..." or "records, which is ..."
    return False


class AppositiveSplit:
    op = "appositive"

    def __init__(self, relatives=True, appositives=True, min_words=MIN_APPOS, version=1):
        self.relatives, self.appositives, self.min_words = relatives, appositives, min_words
        self.version = version  # 2 adds the round-2 skips (_risky) and present-tense definitions

    def _candidates(self, sent):
        out = []
        root = sent.root
        for tok in sent:
            if self.appositives and tok.dep_ == "appos":
                anchor = tok.head
                span = list(tok.subtree)
                kind = "appositive"
            elif self.relatives and tok.dep_ == "relcl":
                anchor = tok.head
                span = list(tok.subtree)
                rel = span[0]
                if rel.lower_ not in ("which", "who") or not any(
                        c.dep_ in ("nsubj", "nsubjpass") and c.i == rel.i for c in tok.children):
                    continue
                kind = "relative"
            else:
                continue
            s, e = span[0].i, span[-1].i
            if anchor.pos_ not in ("NOUN", "PROPN") or anchor.i > s:
                continue
            # The anchor must end right before the comma, and must not be
            # coordinated ("between Forster and Arnold, who ..." is ambiguous).
            own = {t.i for t in anchor.subtree} - {t.i for t in span}
            if any(i not in own for i in range(anchor.i, s - 1)):
                continue
            if anchor.dep_ == "conj" or any(c.dep_ == "conj" for c in anchor.children):
                continue
            if span[0].pos_ in ("ADV", "ADP", "CCONJ", "SCONJ"):
                continue
            if kind == "relative" and rel.lower_ == "who" and anchor.ent_type_ in ("GPE", "LOC", "ORG", "FAC"):
                continue
            if kind == "relative" and not (
                    anchor.dep_ in ("nsubj", "nsubjpass", "dobj", "attr")
                    or anchor.dep_ == "pobj" and anchor.head.head.pos_ in ("VERB", "AUX")):
                continue  # "a version of X, which ..." / "soldiers from X, who ..." are ambiguous
            if not _comma_set(s, e, sent) or _bad_span(span):
                continue
            words = sum(1 for t in span if not t.is_punct)
            if words < self.min_words:
                continue
            if kind == "appositive":
                head = tok
                if head.pos_ not in ("NOUN",) or not any(c.dep_ in ("det", "poss") for c in head.children):
                    continue
                if any(t.dep_ == "relcl" for t in span) and span[-1].i == sent.end - 2 and False:
                    continue
            # Where is the anchor? Subject of the main clause, or the
            # material ends the sentence.
            subject_side = anchor.dep_ in ("nsubj", "nsubjpass") and anchor.head.i == root.i and e < root.i
            final = e == sent.end - 2
            if not (subject_side or final):
                continue
            if self.version >= 2 and _risky(kind, tok, anchor, span, s, sent, final):
                continue
            if anchor.left_edge.i < sent.start or any(t.text == "," for t in sent.doc[anchor.left_edge.i: s - 1]):
                continue
            out.append({"kind": kind, "tok": tok, "anchor": anchor, "start": s, "end": e,
                        "subject_side": subject_side, "final": final})
        return out

    def rewrite_sentence(self, sentence, context=()):
        if any(q in sentence for q in QUOTES):
            return sentence, []
        doc = parse(sentence)
        sents = list(doc.sents)
        if len(sents) != 1:
            return sentence, []
        sent = sents[0]
        for c in self._candidates(sent):
            anchor_text = _anchor_text(c["anchor"], c["start"] - 1)
            if not anchor_text or len(anchor_text.split()) > 10:
                continue
            span = [t for t in doc[c["start"]: c["end"] + 1]]
            # The main sentence without ", <span>" (and its closing comma).
            removed = set(range(c["start"] - 1, c["end"] + 1))
            nxt = doc[c["end"] + 1]
            if nxt.text == "," and not c["final"]:
                removed.add(nxt.i)
            main = Pieces()
            for t in sent:
                if t.i in removed:
                    continue
                main.token(t)
            main_text = main.text()
            if c["kind"] == "appositive":
                pn = number_of(c["anchor"])
                tense = _main_tense(sent)
                if self.version >= 2:
                    # A definition is timeless ("Esomeprazole is an enantiomer
                    # of omeprazole"), unless the appositive marks a past role.
                    tense = "past" if any(t.lower_ in PAST_ROLE for t in span) else "pres"
                be = agree("be", tense, pn)
                extra = f"{upper_first(anchor_text)} {be} {span_text(span)}."
            else:
                clause = span[1:]
                extra = f"{upper_first(anchor_text)} {span_text(clause)}."
            if not main_text.endswith((".", "!", "?")):
                main_text += "."
            out = f"{main_text} {extra}"
            feats = {"kind": c["kind"], "span_len": len(span), "sent_len": len(sent),
                     "subject_side": int(c["subject_side"]), "anchor_len": len(anchor_text.split())}
            return out, [Edit(self.op, c["kind"], sentence, out, feats)]
        return sentence, []
