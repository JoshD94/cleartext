"""Opt-in compound splitting with a dependency parser and source text edits.

No decoder, transformer inference, training or replacement-word generation.
The parser is the existing spaCy en_core_web_sm model. Ambiguous scope abstains.
"""

from ..features import nlp
from ..lexical import MODAL, NEG
from ..protected_literals import literal_spans, preserved_literals
from .common import complete_clause, structure_checks
from .discourse import discourse_checks


SUBJECTS = {"nsubj", "nsubjpass"}
EMBEDDED = {"ccomp", "xcomp", "advcl", "relcl", "acl", "csubj"}


def _capitalized(text):
    # Preserve the rest of the original span, including protocol capitalization.
    return text[:1].upper() + text[1:]


def _finite(predicate):
    return any(
        "Fin" in t.morph.get("VerbForm")
        for t in [predicate, *predicate.children]
        if t == predicate or t.dep_ in {"aux", "auxpass", "cop"}
    )


def _matrix(predicate, root):
    seen = set()
    while predicate.dep_ == "conj" and predicate.i not in seen:
        seen.add(predicate.i)
        predicate = predicate.head
    return predicate == root


class DependencyCompoundSplitter:
    """One accepted boundary per source sentence; defaults remain unchanged."""

    backend = "dependency_compound"
    rule_name = "dependency compound split"
    supports_passages = True

    def validate(self, source, output):
        old, new = nlp()(source), nlp()(output)
        checks = structure_checks(source, output, old, new)
        # Unlike the one-sentence learned adapter, passages may exceed 4 sentences.
        checks["checks"]["sentence_count"] = (
            len(list(old.sents)) <= len(list(new.sents)) <= 2 * len(list(old.sents))
        )
        checks["checks"].update(discourse_checks(old, new))
        checks["checks"]["code_literals"] = preserved_literals(source, output)["pass"]
        checks["failed"] = [k for k, v in checks["checks"].items() if not v]
        checks["pass"] = not checks["failed"]
        return checks

    def _sentence(self, source):
        doc = nlp()(source)
        sentences = list(doc.sents)
        if len(sentences) != 1 or not source.endswith(".") or len(doc) > 120:
            return source, None
        # Quoted speech, parentheticals and protected literals need their own
        # boundary policy. No attempts are made inside them in this version.
        if literal_spans(source) or any(t.is_quote or t.text in {"(", ")", ";", ":"} for t in doc):
            return source, None
        if any(t.lower_ in NEG | MODAL | {"either", "neither", "both"} for t in doc):
            return source, None
        root = sentences[0].root
        for join in doc:
            if join.lower_ not in {"and", "but", "so"} or join.i == 0:
                continue
            is_so = join.lower_ == "so"
            if not is_so and (join.dep_ != "cc" or not _matrix(join.head, root)):
                continue
            # A comma + so can be parsed as a backward ccomp, rather than conj.
            # Accept only this explicit result boundary and independently parse
            # both halves. Never split purpose clauses (so that / so as to).
            if is_so and doc[join.i - 1].text != ",":
                continue
            if join.i + 1 >= len(doc) or doc[join.i + 1].lower_ in {"that", "as"}:
                continue
            embedded = [t for t in doc if t.dep_ in EMBEDDED]
            if embedded and not (
                is_so and len(embedded) == 1 and embedded[0].dep_ == "ccomp"
                and embedded[0].i < join.i < root.i
                and embedded[0].head == root
            ):
                continue
            subject_positions = [t.i for t in doc[:join.i] if t.dep_ in SUBJECTS]
            if not subject_positions or any(
                t.dep_ in {"prep", "advmod", "npadvmod"} and t.i < min(subject_positions)
                for t in doc[:join.i]
            ):
                continue
            # Do not cross remaining predicate coordinations in either half.
            # A later call can tackle another boundary; this pass is bounded.
            boundary = doc[join.i - 1] if doc[join.i - 1].text == "," else join
            left = source[:boundary.idx].rstrip()
            right = source[join.idx + len(join):].strip()
            prefix = ""
            right_predicates = [
                t for t in doc[join.i + 1:]
                if t.pos_ in {"VERB", "AUX"} and t.dep_ == "conj" and _matrix(t, root)
            ]
            explicit = any(
                c.dep_ in SUBJECTS for t in right_predicates for c in t.children
            )
            if not is_so and not explicit:
                # Shared subject: duplicate an existing definite/pronominal/name
                # span only when the second predicate is independently finite.
                # No inherited auxiliary, negation or modifier is reconstructed.
                if len(right_predicates) != 1 or not _finite(right_predicates[0]):
                    continue
                subjects = [t for t in root.children if t.dep_ in SUBJECTS]
                if len(subjects) != 1 or not _finite(root):
                    continue
                subject = subjects[0]
                subject_tokens = list(subject.subtree)
                if subject.pos_ == "PRON" and subject.lower_ not in {"i", "you", "he", "she", "it", "we", "they"}:
                    continue
                if subject.pos_ not in {"PRON", "PROPN"} and not any(
                    t.lower_ in {"the", "this", "that", "these", "those"}
                    for t in subject_tokens
                ):
                    continue
                if any(t.dep_ in {"conj", "preconj", "nummod"} for t in subject_tokens):
                    continue
                if any(t.dep_ in {"aux", "auxpass", "advmod", "npadvmod", "prep"} for t in root.children):
                    continue
                objects = {"dobj", "obj", "iobj", "attr", "oprd"}
                if bool(any(c.dep_ in objects for c in root.children)) != bool(
                    any(c.dep_ in objects for c in right_predicates[0].children)
                ):
                    continue
                if any(c.dep_ in {"advmod", "npadvmod", "prep"} for c in right_predicates[0].children):
                    continue
                first, last = min(t.i for t in subject_tokens), max(t.i for t in subject_tokens)
                if last >= root.i:
                    continue
                prefix = doc[first:last + 1].text + " "
            right = prefix + right
            left_doc, right_doc = nlp()(left + "."), nlp()(_capitalized(right))
            if not all(
                len(list(d.sents)) == 1 and complete_clause(next(iter(d.sents)))
                for d in (left_doc, right_doc)
            ):
                continue
            # Retain contrast and consequence rather than deleting the relation.
            connector = _capitalized(join.text) + " " if join.lower_ != "and" else ""
            if connector and prefix and subject.pos_ != "PROPN" and prefix != "I ":
                right = right[:1].lower() + right[1:]
            output = left + ". " + connector + (_capitalized(right) if not connector else right)
            checks = self.validate(source, output)
            if checks["pass"]:
                return output, {"connector": join.lower_, "copied_subject": prefix.strip(), "before": source, "after": output}
        return source, None

    def predict(self, source):
        doc = nlp()(source)
        edits = []
        # Reverse character edits preserve all inter-sentence whitespace.
        output = source
        for sentence in reversed(list(doc.sents)):
            text = sentence.text
            start = len(text) - len(text.lstrip())
            end = len(text.rstrip())
            replacement, edit = self._sentence(text[start:end])
            if edit:
                output = output[:sentence.start_char + start] + replacement + output[sentence.start_char + end:]
                edits.append(edit)
        checks = self.validate(source, output)
        accepted = checks["pass"]
        return {
            "raw": output, "output": output if accepted else source,
            "checks": checks, "edits": list(reversed(edits)) if accepted else [],
            "candidate_count": len(edits), "semantic_certified": False,
            "reason": "Applied checked dependency boundaries." if accepted and edits else "No safe compound boundary accepted.",
        }
