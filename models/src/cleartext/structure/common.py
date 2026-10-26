"""Shared candidate generation, reference metrics, and structural checks.

Checks are conservative diagnostics, not a semantic equivalence certificate.
All features used at inference depend only on the source and candidate.
"""

import re
from collections import Counter
from functools import lru_cache

import numpy as np
from sacremoses import MosesDetokenizer

from ..features import nlp
from ..lexical import MODAL, NEG, PROTECTED
from ..rewrites import structural


@lru_cache(1)
def detokenizer():
    return MosesDetokenizer(lang="en")


def detokenize(tokens):
    return detokenizer().detokenize(list(tokens))


def normalize(text):
    return " ".join(re.findall(r"\w+|[^\w\s]", text.casefold()))


def metric_tokens(text):
    return normalize(text).split()


def sari(source, output, reference):
    """Single-reference, set-based SARI, with deletion precision, on 0..100.

    Average add F1, keep F1, and delete precision over n-grams 1..4.
    Empty operation denominators score zero, including identity additions.
    This is a reference overlap proxy, not a human simplicity/meaning rating.
    """
    seqs = [metric_tokens(t) for t in (source, output, reference)]
    values = []
    for n in range(1, 5):
        old, new, gold = [set(zip(*(s[i:] for i in range(n)))) for s in seqs]

        def f1(actual, expected):
            return 2 * len(actual & expected) / max(1, len(actual) + len(expected))

        added, wanted_add = new - old, gold - old
        kept, wanted_keep = new & old, gold & old
        deleted, wanted_delete = old - new, old - gold
        values.append(
            (
                f1(added, wanted_add)
                + f1(kept, wanted_keep)
                + len(deleted & wanted_delete) / max(1, len(deleted))
            )
            / 3
        )
    return 100 * float(np.mean(values))


def subject_tokens(doc):
    """A generic copy slot for the root subject, learned use decided by the editor."""
    roots = [t for t in doc if t.dep_ == "ROOT"]
    if not roots:
        return []
    subjects = [t for t in roots[0].children if t.dep_ in {"nsubj", "nsubjpass"}]
    if len(subjects) != 1:
        return []
    # Exclude attached relative clauses and appositives from the copy slot.
    excluded = {
        x.i
        for child in subjects[0].children
        if child.dep_ in {"relcl", "appos", "acl"}
        for x in child.subtree
    }
    return [
        t.text for t in subjects[0].subtree if t.i not in excluded and not t.is_punct
    ]


def _arguments(doc):
    """Position-independent predicate/argument signatures, including passive roles.

    Parser errors and pronouns can cause false rejections. This deliberately
    avoids the lexical guard's equal-token-count assumption.
    """
    result = set()

    def argument(token):
        if token.lower_ in {"who", "which", "that"} and token.head.dep_ == "relcl":
            return token.head.head.lemma_.lower()
        return token.lemma_.lower()

    for predicate in doc:
        if predicate.pos_ not in {"VERB", "AUX", "ADJ"}:
            continue
        passive = any(c.dep_ in {"auxpass", "nsubjpass"} for c in predicate.children)
        children = list(predicate.children)
        # Coordinated predicates can inherit a shared subject.
        if predicate.dep_ == "conj" and not any(
            c.dep_ in {"nsubj", "nsubjpass"} for c in children
        ):
            children += [
                c for c in predicate.head.children if c.dep_ in {"nsubj", "nsubjpass"}
            ]
        for child in children:
            if child.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "iobj"}:
                role = (
                    "patient" if child.dep_ in {"nsubjpass", "dobj", "obj"} else "actor"
                )
                if child.dep_ == "iobj":
                    role = "recipient"
                result.add((predicate.lemma_.lower(), role, argument(child)))
            if passive and child.dep_ == "agent":
                for obj in child.children:
                    if obj.dep_ == "pobj":
                        result.add((predicate.lemma_.lower(), "actor", argument(obj)))
    return result


def _operators(doc):
    """Track which predicates carry negation and modal scope across a split."""
    result = set()
    for predicate in doc:
        if predicate.pos_ not in {"VERB", "AUX", "ADJ"}:
            continue
        governors = [predicate]
        current, seen = predicate, {predicate.i}
        while current.dep_ == "conj" and current.head.i not in seen:
            if any(
                c.dep_ in {"nsubj", "nsubjpass", "aux", "auxpass"}
                for c in current.children
            ):
                break
            current = current.head
            seen.add(current.i)
            governors.append(current)
        for governor in governors:
            for child in governor.children:
                if child.lower_ in NEG or child.lower_ in MODAL:
                    result.add((predicate.lemma_.lower(), child.lower_))
    return result


def complete_clause(sentence):
    """Require a finite main predicate and its own subject.

    A relative-clause verb cannot complete a noun phrase. Participles need a
    finite auxiliary attached to the main predicate. Standalone subordinate
    clauses and imperatives abstain conservatively. Parser errors still apply.
    """
    root = sentence.root
    children = list(root.children)
    if any(t.dep_ == "mark" for t in children):
        return False
    # A detached relative such as "Where she studies networks." still has a
    # finite root and subject. Keep embedded relatives attached to their noun.
    pending = list(children)
    while pending:
        token = pending.pop()
        if token.dep_ in {"relcl", "advcl", "acl", "ccomp", "xcomp", "csubj"}:
            continue
        if token.tag_ in {"WDT", "WP", "WP$", "WRB"}:
            return False
        pending.extend(token.children)
    subjects = [t for t in children if t.dep_ in {"nsubj", "nsubjpass"}]
    expletives = [t for t in children if t.dep_ == "expl"]
    # Coordination has one subject head. Two direct subjects often mean that a
    # copied subject was prepended to a clause which already had one.
    if len(subjects) > 1 or not (subjects or expletives):
        return False
    if any(
        t.dep_ == "dep" and t.pos_ in {"NOUN", "PROPN", "PRON"} and t.i < root.i
        for t in children
    ):
        return False
    # This stage produces declarative prose. A post-verbal subject without an
    # expletive can be a parser's repair of a fragment such as "Is one of ...".
    if not expletives and subjects[0].i > root.i:
        return False
    predicates = [root] + [t for t in children if t.dep_ in {"aux", "auxpass", "cop"}]
    return any(
        t.pos_ in {"VERB", "AUX"} and "Fin" in t.morph.get("VerbForm")
        for t in predicates
    )


def _finite_predicates(doc):
    """Predicates with their own or a coordinated finite auxiliary.

    Splitting must not turn a noun or an unattached participle into a newly
    finite verb just because the parser assigns it a different tag afterward.
    This is a source-relative diagnostic, not an independent grammar judgment.
    """
    result = set()
    for token in doc:
        if token.pos_ not in {"VERB", "AUX", "ADJ"}:
            continue
        current, seen = token, set()
        while current.i not in seen:
            seen.add(current.i)
            predicates = [current] + [
                c for c in current.children if c.dep_ in {"aux", "auxpass", "cop"}
            ]
            if any(
                p.pos_ in {"VERB", "AUX"} and "Fin" in p.morph.get("VerbForm")
                for p in predicates
            ):
                result.add(token.lemma_.lower())
                break
            if current.dep_ != "conj":
                break
            current = current.head
    return result


def structure_checks(source, output, source_doc=None, output_doc=None):
    old = source_doc if source_doc is not None else nlp()(source)
    new = output_doc if output_doc is not None else nlp()(output)
    oc, nc = Counter(t.lower_ for t in old), Counter(t.lower_ for t in new)
    content = lambda d: Counter(
        t.lemma_.lower()
        for t in d
        if t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"} and not t.is_stop
    )
    a, b = content(old), content(new)
    sentences = list(new.sents)
    checks = {
        "numbers": Counter(re.findall(r"\d+(?:[,.]\d+)*", source))
        == Counter(re.findall(r"\d+(?:[,.]\d+)*", output)),
        "negation": all(oc[k] == nc[k] for k in NEG),
        "modality": all(oc[k] == nc[k] for k in MODAL),
        "entities": all(e.text.casefold() in output.casefold() for e in old.ents),
        "technical_phrases": all(
            p not in source.casefold() or p in output.casefold() for p in PROTECTED
        ),
        "content_retained": not (a - b),
        "no_new_content": not (b.keys() - a.keys()),
        "argument_roles": _arguments(old) <= _arguments(new),
        "operator_scope": _operators(old) == _operators(new),
        "finite_predicates": _finite_predicates(new) <= _finite_predicates(old),
        "logical_connectives": all(
            oc[k] == nc[k]
            for k in (
                "if",
                "unless",
                "because",
                "although",
                "while",
                "when",
                "until",
                "but",
                "only",
                "before",
                "after",
            )
        ),
        "sentence_count": 1 <= len(sentences) <= 4,
        "complete_clauses": bool(sentences)
        and all(complete_clause(sent) for sent in sentences),
    }
    failed = [k for k, v in checks.items() if not v]
    return {
        "pass": not failed,
        "checks": checks,
        "failed": failed,
        "content_recall": sum((a & b).values()) / max(1, sum(a.values())),
        "semantic_certified": False,
    }


def _split_text(text, boundary, right_start, prefix=""):
    # Slice the original characters. Detokenization can damage protocol names,
    # hyphens and apostrophes even when a split itself needs no word changes.
    left = text[:boundary].rstrip().rstrip(",;").rstrip()
    right = text[right_start:].lstrip()
    if not left or not right:
        return text
    right = prefix + right
    return left + ". " + right[:1].upper() + right[1:]


def _split_candidates(text, doc):
    """Offer independent clauses or a finite main-verb coordination.

    Never copy a subject into a clause with its own subject, into a noun list,
    or into an embedded clause. Shared auxiliaries and logical conditions
    abstain rather than guessing how to distribute their scope.
    """
    if not text.rstrip().endswith(".") or any(t.is_quote for t in doc):
        return
    if any(
        t.dep_ == "mark"
        or t.lower_ in NEG | MODAL | {"but", "however", "otherwise", "therefore"}
        for t in doc
    ):
        return
    root = list(doc.sents)[0].root
    for token in doc:
        if token.text == ";":
            # Each side must already stand on its own; semicolons never need a
            # copied subject. Re-parsing below also handles parser attachment
            # differences between a semicolon and two separate sentences.
            yield _split_text(text, token.idx, token.idx + 1), "boundary"
            continue
        if token.lower_ != "and" or token.dep_ != "cc" or token.head != root:
            continue
        conjuncts = [
            c
            for c in root.children
            if c.dep_ == "conj" and c.pos_ in {"VERB", "AUX"}
        ]
        if len(conjuncts) != 1:
            continue
        predicate = conjuncts[0]
        right_start = token.i + 1
        if predicate.i < right_start:
            continue
        subtree = {t.i for t in predicate.subtree}
        # No argument/modifier may cross the proposed boundary. In particular,
        # a trailing shared modifier attached to the first predicate stays put.
        if any(
            not t.is_punct and (t.i in subtree) != (t.i >= right_start)
            for t in doc
        ):
            continue
        subjects = [
            c for c in predicate.children if c.dep_ in {"nsubj", "nsubjpass", "expl"}
        ]
        if subjects:
            yield _split_text(text, token.idx, doc[right_start].idx), "boundary"
            continue
        if "Fin" not in predicate.morph.get("VerbForm") or any(
            c.dep_ in {"aux", "auxpass", "neg"} for c in root.children
        ):
            continue
        subjects = [c for c in root.children if c.dep_ == "nsubj"]
        if len(subjects) != 1:
            continue
        subject = subjects[0]
        tokens = list(subject.subtree)
        if any(t.dep_ in {"relcl", "acl", "appos", "conj"} for t in tokens):
            continue
        if tokens[-1].i >= root.i or len(tokens) != tokens[-1].i - tokens[0].i + 1:
            continue
        # "Buys and sells cars" may share the right-hand object. Do not turn it
        # into "Buys. Sells cars" by silently changing the first predicate.
        objects = lambda p: any(c.dep_ in {"dobj", "obj", "attr"} for c in p.children)
        if objects(root) != objects(predicate):
            continue
        prefix = doc[tokens[0].i : tokens[-1].i + 1].text + " "
        yield _split_text(text, token.idx, doc[right_start].idx, prefix), "copy_subject"


def candidate_pool(text, doc=None):
    """Bounded, checked proposals for the learned ranker, including identity."""
    doc = doc if doc is not None else nlp()(text)
    rows = [{"output": text, "kind": "identity"}]
    out, edits = structural(text)
    if edits and structure_checks(text, out, doc)["pass"]:
        rows.append({"output": out, "kind": "existing_rule"})
    if len(list(doc.sents)) != 1:
        return rows
    for out, kind in _split_candidates(text, doc):
        if structure_checks(text, out, doc)["pass"]:
            rows.append({"output": out, "kind": kind})
        if len(rows) >= 25:
            break
    seen = set()
    return [
        r
        for r in rows
        if not (normalize(r["output"]) in seen or seen.add(normalize(r["output"])))
    ]


def candidate_features(source, candidate, source_doc=None, output_doc=None):
    old = source_doc if source_doc is not None else nlp()(source)
    new = output_doc if output_doc is not None else nlp()(candidate["output"])
    lengths = [sum(t.is_alpha for t in sent) for sent in new.sents]
    checks = structure_checks(source, candidate["output"], old, new)
    words = sum(t.is_alpha for t in old)
    return {
        "kind": candidate["kind"],
        "source_words": words,
        "word_ratio": sum(lengths) / max(1, words),
        "sentences": len(lengths),
        "max_length_ratio": max(lengths, default=0) / max(1, words),
        "min_length": min(lengths, default=0),
        "length_variance": float(np.var(lengths)) if lengths else 0,
        "subclauses": sum(t.dep_ in {"relcl", "advcl", "ccomp", "xcomp"} for t in new),
        "dependency_distance": float(np.mean([abs(t.i - t.head.i) for t in new]))
        if len(new)
        else 0,
        "source_dependency_distance": float(np.mean([abs(t.i - t.head.i) for t in old]))
        if len(old)
        else 0,
        "content_recall": checks["content_recall"],
        **{f"check_{k}": float(v) for k, v in checks["checks"].items()},
    }
