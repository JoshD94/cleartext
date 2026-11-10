"""Tag each "and"/"but"/"so"/"yet" as a split point or not, without an LLM.

A logistic-regression tagger over parser and part-of-speech features decides
where to split; `DependencyCompoundSplitter` still makes the text edit and runs
its preservation checks. Labels come from WikiSplit: the connector a person
split at is positive; other connectors are weighted negatives, because a person
writes one split per sentence and an unchosen connector may still be valid.
"""

import io
import zipfile
from pathlib import Path

import joblib

from ..lexical import MODAL, NEG
from .common import detokenize

CONNECTORS = {"and", "but", "so", "yet"}
SUBJECTS = {"nsubj", "nsubjpass", "expl"}
EMBEDDED = {"ccomp", "xcomp", "advcl", "relcl", "acl", "csubj"}


def _bucket(n):
    return str(min(n // 3, 10))


def _matrix(token, root):
    seen = set()
    while token.dep_ == "conj" and token.i not in seen:
        seen.add(token.i)
        token = token.head
    return token == root


def _finite(tokens):
    return any(t.pos_ in {"VERB", "AUX"} and "Fin" in t.morph.get("VerbForm") for t in tokens)


def features(doc, join):
    """Source-only features for one connector token."""
    left, right = doc[:join.i], doc[join.i + 1:]
    prev = doc[join.i - 1]
    nxt = doc[join.i + 1] if join.i + 1 < len(doc) else None
    nxt2 = doc[join.i + 2] if join.i + 2 < len(doc) else None
    root = join.sent.root
    head = join.head
    crossing = inside = False
    for t in doc:
        if t.dep_ not in EMBEDDED:
            continue
        span = [x.i for x in t.subtree]
        if min(span) < join.i < max(span):
            inside = True
        elif (t.head.i > join.i) != (min(span) > join.i):
            # The clause sits on one side but attaches to the other.
            crossing = True
    words = lambda span: sum(not t.is_punct and not t.is_space for t in span)
    return {
        "conn=" + join.lower_: 1,
        "comma_before": float(prev.text == ","),
        "prev_tag=" + prev.tag_: 1,
        "next_tag=" + (nxt.tag_ if nxt else "END"): 1,
        "next_word=" + (nxt.lower_ if nxt is not None and nxt.is_stop else "OTHER"): 1,
        "next2_tag=" + (nxt2.tag_ if nxt2 else "END"): 1,
        "dep=" + join.dep_: 1,
        "head_pos=" + head.pos_: 1,
        "head_dep=" + head.dep_: 1,
        "conn+comma+next=" + join.lower_ + str(prev.text == ",") + (nxt.pos_ if nxt else ""): 1,
        "matrix_coordination": float(join.dep_ == "cc" and _matrix(head, root)),
        "right_subject": float(any(t.dep_ in SUBJECTS and t.head.i > join.i for t in right)),
        "left_subject": float(any(t.dep_ in SUBJECTS for t in left)),
        "right_finite": float(_finite(right)),
        "left_finite": float(_finite(left)),
        "embedded_crossing": float(crossing),
        "inside_embedded": float(inside),
        "left_words=" + _bucket(words(left)): 1,
        "right_words=" + _bucket(words(right)): 1,
        "connectors=" + str(min(4, sum(t.lower_ in CONNECTORS for t in doc))): 1,
        "neg_left": float(any(t.lower_ in NEG for t in left)),
        "neg_right": float(any(t.lower_ in NEG for t in right)),
        "modal_left": float(any(t.lower_ in MODAL for t in left)),
        "modal_right": float(any(t.lower_ in MODAL for t in right)),
        "leading_modifier": float(any(t.dep_ in {"prep", "advmod", "npadvmod", "advcl"}
                                      and t.head == root and t.i < root.i for t in left)),
    }


def wikisplit_label(complex_tokens, simple1, simple2):
    """Occurrence number of the connector a person split at, else None.

    Clean only: the first reference sentence equals the text before the
    connector and the second ends with the text after it, with at most the
    connector, "however" and six subject words added in front.
    """
    first = [t.lower() for t in simple1 if t != "."]
    second = [t.lower() for t in simple2]
    lowered = [t.lower() for t in complex_tokens]
    occurrence = -1
    for k, token in enumerate(lowered):
        if token not in CONNECTORS:
            continue
        occurrence += 1
        if k == 0:
            continue
        left = lowered[:k]
        if left and left[-1] == ",":
            left = left[:-1]
        right = lowered[k + 1:]
        if left != first or len(right) < 3 or second[-len(right):] != right:
            continue
        prefix = second[:len(second) - len(right)]
        if prefix[:1] and prefix[0] in CONNECTORS | {"however", ","}:
            prefix = [t for t in prefix[1:] if t != ","]
        if len(prefix) <= 6:
            return occurrence
    return None


def wikisplit_rows(path, limit=0, stride=1):
    """Connector sentences from a WikiSplit tsv or zipped tsv, every `stride`th line."""
    path = Path(path)
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        handle = io.TextIOWrapper(archive.open(archive.namelist()[0]), encoding="utf-8")
    else:
        handle = path.open(encoding="utf-8")
    with handle:
        return _rows(handle, limit, stride)


def _rows(handle, limit, stride):
    rows = []
    for n, line in enumerate(handle):
        if n % stride:
            continue
        line = line.rstrip("\n")
        complex_text, _, simple = line.partition("\t")
        parts = simple.split(" <::::> ")
        tokens = complex_text.split()
        if len(parts) != 2 or not CONNECTORS & {t.lower() for t in tokens}:
            continue
        s1, s2 = parts[0].split(), parts[1].split()
        rows.append({
            "source": detokenize(tokens),
            "reference": detokenize(s1) + " " + detokenize(s2),
            "reference_first": detokenize(s1),
            "connector_occurrence": wikisplit_label(tokens, s1, s2),
        })
        if limit and len(rows) >= limit:
            break
    return rows


class BoundaryTagger:
    """Probability that a person would split at a connector."""

    def __init__(self, vectorizer, model, threshold):
        self.vectorizer, self.model, self.threshold = vectorizer, model, threshold

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        return cls(data["vectorizer"], data["model"], data["threshold"])

    def save(self, path, **metadata):
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model,
                     "threshold": self.threshold, **metadata}, path)

    def score(self, doc, join):
        return float(self.model.predict_proba(self.vectorizer.transform([features(doc, join)]))[0, 1])
