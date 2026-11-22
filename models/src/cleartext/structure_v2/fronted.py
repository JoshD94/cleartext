"""Long fronted subordinate clauses (structure v2, opt-in).

Two rules, measured separately:

- split ("fronted_split"): a fronted causal, concessive or "after" clause
  becomes its own sentence and the main clause keeps the relation with a
  plain connective (Siddharthan-style discourse rules):
      "Because X, Y."  -> "X. So Y."
      "Although X, Y." -> "X. But Y."
      "After X, Y."    -> "X. Then Y."
- move ("fronted_move"): a long fronted conditional or temporal clause goes
  after the main clause, main idea first:
      "If X, Y."   -> "Y if X."
      "When X, Y." -> "Y when X."

The subordinate clause must be finite with its own subject, and the main
clause must have a subject. Ambiguous markers ("since", "while", "as") and
non-finite clauses ("Based on X, ...") are left alone.
"""

from .common import parse, Edit, Pieces, lower_first, span_text, upper_first

SPLIT = {"because": "So", "although": "But", "though": "But", "even though": "But", "after": "Then"}
MOVE = {"if", "when", "whenever", "unless", "once", "until", "before"}
MIN_CLAUSE = 6  # words in the fronted clause


def _marker(sent):
    first = sent[0]
    two = f"{first.lower_} {sent[1].lower_}" if len(sent) > 1 else ""
    if two in SPLIT:
        return two, 2
    return first.lower_, 1


def _fronted(sent):
    """(advcl head, end index of the clause, marker, marker length) or None."""
    marker, n = _marker(sent)
    if marker not in SPLIT and marker not in MOVE:
        return None
    root = sent.root
    marker_tok = sent[n - 1]
    if marker_tok.dep_ not in ("mark", "advmod") or marker_tok.head.dep_ != "advcl" \
            or marker_tok.head.head.i != root.i:
        return None
    for c in root.children:
        if c.dep_ == "advcl" and c.left_edge.i == sent.start and c.i < root.i:
            end = c.right_edge.i
            comma = sent.doc[end + 1] if end + 1 < sent.end else None
            if comma is None or comma.text != ",":
                return None
            if not any(g.dep_ in ("nsubj", "nsubjpass", "expl") for g in c.children):
                return None
            if c.tag_ not in ("VBD", "VBZ", "VBP", "MD") and not any(
                    g.dep_ in ("aux", "auxpass") and g.tag_ in ("VBD", "VBZ", "VBP", "MD") for g in c.children):
                return None
            if not any(g.dep_ in ("nsubj", "nsubjpass", "expl") for g in root.children):
                return None
            return c, end, marker, n
    return None


class FrontedClause:
    op = "fronted"

    def __init__(self, rules=("split", "move"), min_clause=MIN_CLAUSE):
        unknown = set(rules) - {"split", "move"}
        if unknown:
            raise ValueError(f"Unknown fronted rules: {sorted(unknown)}")
        self.rules, self.min_clause = tuple(rules), min_clause

    def rewrite_sentence(self, sentence, context=()):
        if any(q in sentence for q in ('"', "“", "”")):
            return sentence, []
        doc = parse(sentence)
        sents = list(doc.sents)
        if len(sents) != 1:
            return sentence, []
        sent = sents[0]
        found = _fronted(sent)
        if found is None:
            return sentence, []
        head, end, marker, n = found
        clause = list(doc[sent.start + n: end + 1])
        if sum(1 for t in clause if not t.is_punct) < self.min_clause:
            return sentence, []
        if any(t.text in (";", ":", "(", ")") for t in clause):
            return sentence, []
        main = [t for t in sent if t.i > end + 1]
        if not main or not main[-1].is_punct:
            return sentence, []
        if main[0].lower_ in ("then", "so", "thus", "therefore", "still", "yet"):
            return sentence, []
        final = main[-1].text
        if final not in (".", "!"):
            return sentence, []
        body = main[:-1]
        if not body:
            return sentence, []
        main_text = span_text(body)
        clause_text = span_text(clause)
        feats = {"marker": marker, "clause_len": len(clause), "main_len": len(body), "sent_len": len(sent)}
        if marker in SPLIT and "split" in self.rules:
            first = upper_first(clause_text)
            second = f"{SPLIT[marker]}, {lower_first(main_text, body[0])}" if marker == "after" else \
                f"{SPLIT[marker]} {lower_first(main_text, body[0])}"
            if marker == "after":
                second = f"Then {lower_first(main_text, body[0])}"
            out = f"{first}. {second}{final}"
            return out, [Edit(self.op, "split", sentence, out, {**feats, "rule": "split"})]
        if marker in MOVE and "move" in self.rules:
            # A pronoun subject in the main clause usually points forward to
            # the fronted clause ("If DOE lacks data, it should ..."); moving
            # the clause after it would leave the pronoun without a referent.
            subj = [c for c in sent.root.children if c.dep_ in ("nsubj", "nsubjpass", "expl")]
            if any(t.tag_ in ("PRP", "DT") for t in subj) or any(t.text in (",", "(", ";") for t in body):
                return sentence, []
            if any(t.lower_ in ("however", "therefore", "thus", "though", "instead") for t in clause):
                return sentence, []
            lead = sent[0].text.lower() if not (sent[0].text.isupper() and len(sent[0].text) > 1) else sent[0].text
            sub = span_text(list(doc[sent.start: end + 1]))
            sub = lead + sub[len(sent[0].text):]
            out = f"{upper_first(main_text)} {sub}{final}"
            return out, [Edit(self.op, "move", sentence, out, {**feats, "rule": "move"})]
        return sentence, []
