"""Opt-in compound splitting with a dependency parser and source text edits.

No decoder, transformer inference, training or replacement-word generation.
The parser is the existing spaCy en_core_web_sm model. Ambiguous scope abstains.
"""

from collections import Counter
from pathlib import Path

from ..features import nlp
from ..lexical import MODAL, NEG
from ..protected_literals import literal_spans, preserved_literals
from .boundary_tagger import BoundaryTagger
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


def _subject_groups(doc):
    """Keep joint subjects together, including subjects inherited by conjuncts.

    This closes one known gap in the shared argument-role check. It is still
    parser-dependent and is not a general semantic equivalence test.
    """
    groups = Counter()
    for predicate in doc:
        if predicate.pos_ not in {"VERB", "AUX", "ADJ"}:
            continue
        governor, seen = predicate, set()
        while governor.i not in seen:
            seen.add(governor.i)
            subjects = [c for c in governor.children if c.dep_ in SUBJECTS]
            if subjects or governor.dep_ != "conj":
                break
            governor = governor.head
        for subject in subjects:
            members, pending = [], [subject]
            while pending:
                member = pending.pop()
                members.append(member.lemma_.lower())
                pending.extend(c for c in member.children if c.dep_ == "conj")
            groups[(predicate.lemma_.lower(), subject.dep_, tuple(sorted(members)))] += 1
    return groups


def _enclosed(doc, source):
    """Character ranges inside quotes, brackets or code literals.

    Returns None when quotes or brackets are unbalanced, so the caller abstains.
    """
    ranges = [(span.start, span.end) for span in literal_spans(source)]
    opened, quote = [], None
    for t in doc:
        if t.text in {"(", "["}:
            opened.append(t.idx)
        elif t.text in {")", "]"}:
            if not opened:
                return None
            ranges.append((opened.pop(), t.idx + 1))
        elif t.is_quote and t.tag_ != "POS":
            # Possessive apostrophes ("Jardines' fleet") are not quotes.
            if quote is None:
                quote = t.idx
            else:
                ranges.append((quote, t.idx + 1))
                quote = None
    if opened or quote is not None:
        return None
    return ranges


def _relative_attachments(doc):
    return Counter(
        (t.dep_, t.lemma_.lower(), t.head.lemma_.lower())
        for t in doc if t.dep_ in {"relcl", "acl"}
    )


class DependencyCompoundSplitter:
    """One accepted boundary per source sentence; defaults remain unchanged."""

    backend = "dependency_compound"
    rule_name = "dependency compound split"
    supports_passages = True

    def __init__(self, min_sentence_words=0, min_clause_words=0, guards="scoped", tagger=None):
        # "strict" replays the November 2 behaviour: any quote, bracket,
        # negation or modal in a sentence vetoes every boundary in it.
        # "scoped" only vetoes boundaries inside quotes/brackets/literals and
        # leaves negation/modal scope to the operator and count checks.
        if guards not in {"scoped", "strict"}:
            raise ValueError("guards must be 'scoped' or 'strict'")
        self.guards = guards
        # Optional BoundaryTagger (or path to one): a trained tagger replaces the
        # hand-written matrix/relative-clause vetoes. Preservation checks stay.
        if isinstance(tagger, (str, Path)):
            tagger = BoundaryTagger.load(tagger)
        if tagger is not None and guards != "scoped":
            raise ValueError("a tagger requires scoped guards")
        self.tagger = tagger
        for name, value in (("min_sentence_words", min_sentence_words),
                            ("min_clause_words", min_clause_words)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.min_sentence_words = min_sentence_words
        self.min_clause_words = min_clause_words

    def validate(self, source, output):
        old, new = nlp()(source), nlp()(output)
        checks = structure_checks(source, output, old, new)
        # Unlike the one-sentence learned adapter, passages may exceed 4 sentences.
        checks["checks"]["sentence_count"] = (
            len(list(old.sents)) <= len(list(new.sents)) <= 2 * len(list(old.sents))
        )
        # An untouched footnote/fragment elsewhere must not veto a checked split.
        # Match exact text AND multiplicity: no new, edited or duplicated fragment
        # is allowed. Sentence-level proposals still have to be complete clauses.
        old_fragments = Counter(s.text.strip() for s in old.sents if not complete_clause(s))
        new_fragments = Counter(s.text.strip() for s in new.sents if not complete_clause(s))
        checks["checks"]["complete_clauses"] = old_fragments == new_fragments
        checks["checks"].update(discourse_checks(old, new))
        checks["checks"]["code_literals"] = preserved_literals(source, output)["pass"]
        checks["checks"]["subject_groups"] = _subject_groups(old) == _subject_groups(new)
        checks["checks"]["relative_attachments"] = _relative_attachments(old) == _relative_attachments(new)
        checks["failed"] = [k for k, v in checks["checks"].items() if not v]
        checks["pass"] = not checks["failed"]
        return checks

    def _sentence(self, source, diagnostics=None):
        diagnostics = diagnostics if diagnostics is not None else []
        doc = nlp()(source)
        joins = [t for t in doc if t.lower_ in {"and", "but", "so"} and t.i > 0]
        word_count = sum(not t.is_punct and not t.is_space for t in doc)

        def rejected_sentence(reason):
            diagnostics.extend({"connector": t.lower_, "offset": t.idx,
                                "sentence_words": word_count,
                                "accepted": False, "reason": reason} for t in joins)
            return source, None

        sentences = list(doc.sents)
        if len(sentences) != 1 or not source.endswith(".") or len(doc) > 120:
            return rejected_sentence("unsupported_sentence_shape")
        if word_count < self.min_sentence_words:
            return rejected_sentence("sentence_too_short")
        # Quoted speech, parentheticals and protected literals need their own
        # boundary policy. Sentences containing them still abstain entirely.
        enclosed = []
        if self.guards == "strict":
            if literal_spans(source) or any(t.is_quote or t.text in {"(", ")", ";", ":"} for t in doc):
                return rejected_sentence("literal_quote_or_parenthetical")
            if any(t.lower_ in NEG | MODAL | {"either", "neither", "both"} for t in doc):
                return rejected_sentence("negation_modality_or_correlative")
        else:
            # Semicolons and colons already mark a clause boundary whose scope
            # this stage does not model.
            enclosed = _enclosed(doc, source)
            if enclosed is None or any(t.text in {";", ":"} for t in doc):
                return rejected_sentence("literal_quote_or_parenthetical")
        root = sentences[0].root
        for join in joins:
            detail = {"connector": join.lower_, "offset": join.idx,
                      "sentence_words": word_count, "accepted": False}
            diagnostics.append(detail)
            if any(start <= join.idx < end for start, end in enclosed):
                detail["reason"] = "literal_quote_or_parenthetical"
                continue
            # A correlative before the connector ("both A and B", "either ...")
            # binds the two halves together.
            if any(t.lower_ in {"either", "neither", "both"} for t in doc[:join.i]):
                detail["reason"] = "negation_modality_or_correlative"
                continue
            is_so = join.lower_ == "so"
            # The tagger only adds boundaries: below its threshold the
            # hand-written vetoes decide, exactly as without a tagger.
            tagged = False
            if self.tagger is not None:
                detail["tagger_score"] = round(self.tagger.score(doc, join), 4)
                tagged = detail["tagger_score"] >= self.tagger.threshold
            # Tagged boundaries may join embedded-level predicates, never nouns.
            if not is_so and (join.dep_ != "cc" or (not tagged and not _matrix(join.head, root))
                              or join.head.pos_ not in {"VERB", "AUX", "ADJ"}):
                detail["reason"] = "not_matrix_coordination"
                continue
            # A comma + so can be parsed as a backward ccomp, rather than conj.
            # Accept only this explicit result boundary and independently parse
            # both halves. Never split purpose clauses (so that / so as to).
            if is_so and doc[join.i - 1].text != ",":
                detail["reason"] = "so_requires_comma"
                continue
            if join.i + 1 >= len(doc) or doc[join.i + 1].lower_ in {"that", "as"}:
                detail["reason"] = "purpose_or_missing_clause"
                continue
            boundary = doc[join.i - 1] if doc[join.i - 1].text == "," else join
            embedded = [t for t in doc if t.dep_ in EMBEDDED]
            backward_so = (
                is_so and len(embedded) == 1 and embedded[0].dep_ == "ccomp"
                and embedded[0].i < join.i < root.i
                and embedded[0].head == root
            )
            # A relative must remain wholly within one half, with its noun.
            # Other embedded clauses can carry shared scope; keep abstaining.
            local_relatives = bool(embedded) and all(
                t.dep_ in {"relcl", "acl"}
                and (all(x.i < boundary.i for x in [t.head, *t.subtree])
                     or all(x.i > join.i for x in [t.head, *t.subtree]))
                for t in embedded
            )
            if tagged:
                # Any embedded clause must sit wholly on its head's side, and
                # a leading adverbial clause ("If X, A and B") scopes over both.
                local_relatives = bool(embedded) and all(
                    t.dep_ in {"relcl", "acl"} for t in embedded)
                if any(
                    (min(x.i for x in t.subtree) < join.i) != (max(x.i for x in t.subtree) < join.i)
                    or (t.head.i < join.i) != (t.i < join.i)
                    or (t.dep_ == "advcl" and t.i < t.head.i)
                    for t in embedded
                ):
                    detail["reason"] = "embedded_scope_crosses_or_is_unsupported"
                    continue
            elif embedded and not (backward_so or local_relatives):
                detail["reason"] = "embedded_scope_crosses_or_is_unsupported"
                continue
            subject_positions = [t.i for t in doc[:join.i] if t.dep_ in SUBJECTS]
            # Only sentence-level modifiers scope over both halves; an adverb
            # inside the subject ("Almost all children") does not.
            if not subject_positions or any(
                t.dep_ in {"prep", "advmod", "npadvmod"} and t.i < min(subject_positions)
                and (t.head == root or not tagged)
                for t in doc[:join.i]
            ):
                detail["reason"] = "leading_modifier_scope"
                continue
            # Try at most one accepted boundary per sentence in this pass.
            left = source[:boundary.idx].rstrip()
            right = source[join.idx + len(join):].strip()
            left_words = sum(not t.is_punct and not t.is_space for t in doc[:boundary.i])
            right_words = sum(not t.is_punct and not t.is_space for t in doc[join.i + 1:])
            detail.update(left_words=left_words, right_words=right_words)
            if min(left_words, right_words) < self.min_clause_words:
                detail["reason"] = "clause_too_short"
                continue
            prefix = ""
            right_predicates = [
                t for t in doc[join.i + 1:]
                if t.pos_ in {"VERB", "AUX"} and t.dep_ == "conj"
                and (_matrix(t, root) if not tagged else t.head.i < join.i)
            ]
            explicit = any(
                c.dep_ in SUBJECTS for t in right_predicates for c in t.children
            )
            if not tagged and local_relatives and (
                is_so or len(right_predicates) != 1 or not explicit
                or not any(c.dep_ in SUBJECTS for c in root.children)
                or not _finite(root) or not _finite(right_predicates[0])
            ):
                detail["reason"] = "relative_requires_independent_subjects"
                continue
            if not is_so and not explicit:
                detail["reason"] = "unsafe_shared_subject_or_predicate"
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
                if tagged:
                    # Copy only the subject's core noun phrase, never its
                    # relative clause, appositive or parenthetical.
                    core, start = [], subject.i
                    for t in reversed(list(subject.lefts)):
                        if t.dep_ not in {"det", "compound", "amod", "poss"} or t.right_edge.i != start - 1:
                            break
                        core.append(t)
                        start = t.left_edge.i
                    # Never copy part of a hyphenated or attached name.
                    if start > 0 and (not doc[start - 1].whitespace_ or doc[start - 1].text == "-"):
                        continue
                    subject_tokens = list(doc[start:subject.i + 1])
                if any(t.dep_ in {"conj", "preconj", "nummod"} for t in subject_tokens):
                    continue
                # Untagged: no modifier on the first predicate at all. Tagged:
                # only modifiers before it, which could scope over both verbs.
                if any(t.dep_ in {"aux", "auxpass", "advmod", "npadvmod", "prep"}
                       and (not tagged or (t.i < root.i and t.dep_ not in {"aux", "auxpass"}))
                       for t in root.children):
                    continue
                objects = {"dobj", "obj", "iobj", "attr", "oprd"}
                if bool(any(c.dep_ in objects for c in root.children)) != bool(
                    any(c.dep_ in objects for c in right_predicates[0].children)
                ):
                    continue
                # Tagged: a modifier before the second verb would sit between
                # the copied subject and its verb ("He from 1954 presented").
                if any(c.dep_ in {"advmod", "npadvmod", "prep"} and (not tagged or c.i < right_predicates[0].i)
                       for c in right_predicates[0].children):
                    continue
                # A relative clause ending at the connector may own the second
                # verb, so the root subject is not safely the one to copy.
                if any(t.dep_ in {"relcl", "acl"} and root.i < t.i < join.i for t in doc):
                    continue
                first, last = min(t.i for t in subject_tokens), max(t.i for t in subject_tokens)
                if last >= root.i:
                    continue
                prefix = doc[first:last + 1].text + " "
            # A second half that opens with its verb has no subject of its own,
            # however the parser labels a later reflexive ("But was himself ...").
            if not prefix and join.i + 1 < len(doc) and doc[join.i + 1].pos_ in {"AUX", "VERB"}:
                detail["reason"] = "incomplete_clause"
                continue
            right = prefix + right
            left_doc, right_doc = nlp()(left + "."), nlp()(_capitalized(right))
            if not all(
                len(list(d.sents)) == 1 and complete_clause(next(iter(d.sents)))
                for d in (left_doc, right_doc)
            ):
                detail["reason"] = "incomplete_clause"
                continue
            # The first half must not end in an unfinished coordination
            # ("caught by the Zhus but Yang Xiong.").
            # A contrastive connector between nouns is almost always a parse
            # error that hides a clause boundary.
            if any(t.dep_ == "cc" and not any(c.dep_ == "conj" for c in t.head.children) for t in left_doc) or (
                tagged and any(t.lower_ in {"but", "yet"} and t.dep_ == "cc"
                               and t.head.pos_ in {"NOUN", "PROPN", "PRON"} for t in doc[:join.i])):
                detail["reason"] = "incomplete_clause"
                continue
            # Retain contrast and consequence rather than deleting the relation.
            connector = _capitalized(join.text) + " " if join.lower_ != "and" else ""
            if connector and prefix and subject.pos_ != "PROPN" and prefix != "I ":
                right = right[:1].lower() + right[1:]
            output = left + ". " + connector + (_capitalized(right) if not connector else right)
            checks = self.validate(source, output)
            if checks["pass"]:
                detail.update(accepted=True, reason="accepted", copied_subject=prefix.strip())
                return output, {"connector": join.lower_, "copied_subject": prefix.strip(), "before": source, "after": output}
            detail.update(reason="preservation_check_failed", failed_checks=checks["failed"])
        return source, None

    def predict(self, source):
        doc = nlp()(source)
        edits = []
        boundaries = []
        # Reverse character edits preserve all inter-sentence whitespace.
        output = source
        for sentence in reversed(list(doc.sents)):
            text = sentence.text
            start = len(text) - len(text.lstrip())
            end = len(text.rstrip())
            diagnostics = []
            replacement, edit = self._sentence(text[start:end], diagnostics)
            for detail in diagnostics:
                detail["offset"] += sentence.start_char + start
            boundaries.extend(diagnostics)
            if edit:
                output = output[:sentence.start_char + start] + replacement + output[sentence.start_char + end:]
                edits.append(edit)
        checks = self.validate(source, output)
        accepted = checks["pass"]
        if not accepted:
            for detail in boundaries:
                if detail["accepted"]:
                    detail.update(accepted=False, reason="passage_rollback", failed_checks=checks["failed"])
        return {
            "raw": output, "output": output if accepted else source,
            "checks": checks, "edits": list(reversed(edits)) if accepted else [],
            "candidate_count": len(boundaries), "accepted_count": len(edits) if accepted else 0,
            "boundaries": sorted(boundaries, key=lambda d: d["offset"]),
            "length_policy": {"min_sentence_words": self.min_sentence_words,
                              "min_clause_words": self.min_clause_words},
            "semantic_certified": False,
            "reason": "Applied checked dependency boundaries." if accepted and edits else "No safe compound boundary accepted.",
        }
