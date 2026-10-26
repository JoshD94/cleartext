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
        and all(
            any(t.dep_ in {"nsubj", "nsubjpass", "expl"} for t in sent)
            and any(t.pos_ in {"VERB", "AUX"} for t in sent)
            for sent in sentences
        ),
    }
    failed = [k for k, v in checks.items() if not v]
    return {
        "pass": not failed,
        "checks": checks,
        "failed": failed,
        "content_recall": sum((a & b).values()) / max(1, sum(a.values())),
        "semantic_certified": False,
    }


def candidate_pool(text, doc=None):
    """Bounded rule candidates for the learned ranker, including identity.

    Broad punctuation/conjunction proposals deliberately include bad splits.
    The ranker learns preferences from reference overlap; checks reject fragments.
    """
    doc = doc if doc is not None else nlp()(text)
    rows = [{"output": text, "kind": "identity"}]
    out, edits = structural(text)
    if edits:
        rows.append({"output": out, "kind": "existing_rule"})
    if len(list(doc.sents)) != 1:
        return rows
    subject = subject_tokens(doc)
    for i, token in enumerate(doc):
        if token.text not in {",", ";"} and token.lower_ not in {"and", "but"}:
            continue
        left = [t.text for t in doc[:i]]
        right = [t.text for t in doc[i + 1 :]]
        while left and left[-1] in {",", ";"}:
            left.pop()
        if right and right[0].lower() in {"and", "but"}:
            right = right[1:]
        if sum(w.isalpha() for w in left) < 4 or sum(w.isalpha() for w in right) < 4:
            continue
        for prefix, kind in [([], "boundary"), (subject, "copy_subject")]:
            if kind == "copy_subject" and not prefix:
                continue
            tokens = left + ["."] + prefix + right
            out = detokenize(tokens)
            out = re.sub(r"([.!?]\s+)([a-z])", lambda m: m[1] + m[2].upper(), out)
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
