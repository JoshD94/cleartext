"""Empty frames (structure v2, opt-in).

- existential: "There are many people who live here." -> "Many people live here."
  "There were five people killed." -> "Five people were killed."
- cleft: "It was the board that approved the plan." -> "The board approved the plan."

Only subject relatives are removed (the relative pronoun must be the
clause's subject), so the noun phrase can simply take its place. "There is
no evidence that ..." (a content clause) and "It is important that ..."
(extraposition) are left alone.
"""

from .common import parse, Edit, Pieces, agree, lower_first, number_of, span_text, upper_first

REL = {"who", "that", "which"}


class EmptyFrames:
    op = "frames"

    def __init__(self, rules=("existential", "cleft")):
        self.rules = tuple(rules)

    def _existential(self, sent):
        first = sent[0]
        if first.lower_ != "there" or first.dep_ != "expl":
            return None
        be = first.head
        if be.lemma_ != "be" or be.i != first.i + 1 and not (be.i == first.i + 2 and sent[1].dep_ == "aux"):
            return None
        if any(c.dep_ == "neg" for c in be.children):
            return None
        attr = next((c for c in be.children if c.dep_ in ("attr", "nsubj") and c.i > be.i), None)
        if attr is None or attr.pos_ not in ("NOUN", "PROPN", "PRON", "NUM"):
            return None
        det = [c for c in attr.children if c.dep_ == "det"]
        if det and det[0].lower_ in ("no",):
            return None
        rel = next((c for c in attr.children if c.dep_ in ("relcl", "acl") and c.i > attr.i), None)
        if rel is None:
            return None
        rest_after = [t for t in sent if t.i > rel.right_edge.i and not t.is_punct]
        if rest_after:
            return None
        if rel.dep_ == "relcl":
            pron = rel.left_edge
            if pron.lower_ not in REL or not any(c.i == pron.i and c.dep_ in ("nsubj", "nsubjpass")
                                                 for c in rel.children):
                return None
            np = list(sent.doc[attr.left_edge.i: rel.left_edge.i])
            clause = list(sent.doc[rel.left_edge.i + 1: rel.right_edge.i + 1])
            if not clause:
                return None
            return np, clause, None, "relative"
        # Reduced relative: "There were five people killed" -> "Five people were killed".
        if rel.tag_ not in ("VBN", "VBG") or rel.left_edge.i != rel.i:
            return None
        np = list(sent.doc[attr.left_edge.i: rel.i])
        clause = list(sent.doc[rel.i: rel.right_edge.i + 1])
        tense = "past" if be.tag_ == "VBD" else "pres"
        return np, clause, agree("be", tense, number_of(attr)), "reduced"

    def _cleft(self, sent):
        first = sent[0]
        if first.lower_ != "it" or len(sent) < 5:
            return None
        be = first.head
        if be.lemma_ != "be" or be.i != first.i + 1 or be.dep_ != "ROOT":
            return None
        attr = next((c for c in be.children if c.dep_ == "attr" and c.i > be.i), None)
        if attr is None or attr.pos_ not in ("NOUN", "PROPN", "PRON"):
            return None
        rel = next((c for c in list(attr.children) + list(be.children)
                    if c.dep_ in ("relcl", "ccomp") and c.i > attr.i), None)
        if rel is None:
            return None
        pron = rel.left_edge
        if pron.lower_ not in ("who", "that") or pron.i != attr.right_edge.i + 1 and \
                pron.i != max(t.i for t in attr.subtree if t.i < pron.i) + 1:
            return None
        if any(c.dep_.startswith("nsubj") and c.i != pron.i for c in rel.children):
            return None  # the clause has its own subject: not a subject cleft
        if not any(c.dep_.startswith("nsubj") and c.i == pron.i for c in rel.children):
            return None
        np = list(sent.doc[be.i + 1: pron.i])
        clause = list(sent.doc[pron.i + 1: rel.right_edge.i + 1])
        rest_after = [t for t in sent if t.i > rel.right_edge.i and not t.is_punct]
        if rest_after or not clause or not np:
            return None
        return np, clause, None, "cleft"

    def rewrite_sentence(self, sentence, context=()):
        if any(q in sentence for q in ('"', "“", "”")):
            return sentence, []
        doc = parse(sentence)
        sents = list(doc.sents)
        if len(sents) != 1:
            return sentence, []
        sent = sents[0]
        found = None
        if "existential" in self.rules:
            found = self._existential(sent)
        if found is None and "cleft" in self.rules:
            found = self._cleft(sent)
        if found is None:
            return sentence, []
        np, clause, be, rule = found
        final = sent[-1].text if sent[-1].is_punct else "."
        np_text = upper_first(span_text(np))
        middle = f" {be}" if be else ""
        out = f"{np_text}{middle} {span_text(clause)}{final}"
        feats = {"rule": rule, "np_len": len(np), "clause_len": len(clause), "sent_len": len(sent)}
        return out, [Edit(self.op, rule, sentence, out, feats)]
