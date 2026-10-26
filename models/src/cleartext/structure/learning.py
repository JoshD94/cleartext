"""A reference-trained rewrite ranker and a learned monotonic CRF edit tagger."""

import json
import pickle
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from ..features import nlp
from .common import (
    candidate_features,
    candidate_pool,
    detokenize,
    normalize,
    sari,
    structure_checks,
    subject_tokens,
)
from .corpus import write_json

RANKER_FEATURE_VERSION = 2


def parse_sources(data):
    return {
        split: list(nlp().pipe([r["source"] for r in rows], batch_size=64))
        for split, rows in data.items()
    }


class RewriteRanker:
    def __init__(self, model, threshold=0.0):
        self.model, self.threshold = model, threshold
        self.feature_version = RANKER_FEATURE_VERSION

    def predict(self, source, doc=None):
        if getattr(self, "feature_version", None) != RANKER_FEATURE_VERSION:
            raise ValueError(
                "Ranker candidate/check features changed. Refit the ranker with this "
                "source version, or use the original artifact's frozen source."
            )
        doc = doc if doc is not None else nlp()(source)
        candidates = candidate_pool(source, doc)
        features = [candidate_features(source, c, doc) for c in candidates]
        scores = self.model.predict(features)
        baseline = float(scores[0])
        best = max(range(len(candidates)), key=lambda i: float(scores[i]))
        raw = candidates[best]["output"]
        gain = float(scores[best]) - baseline
        checks = structure_checks(source, raw, doc)
        accepted = best != 0 and gain > self.threshold and checks["pass"]
        # Choose among passing candidates, rather than letting the first bad one hide them.
        for i in sorted(range(1, len(candidates)), key=lambda i: -float(scores[i])):
            if float(scores[i]) - baseline <= self.threshold:
                break
            if structure_checks(source, candidates[i]["output"], doc)["pass"]:
                return {
                    "raw": raw,
                    "output": candidates[i]["output"],
                    "score": float(scores[i]) - baseline,
                    "candidate_count": len(candidates),
                    "raw_checks": checks,
                }
        return {
            "raw": raw,
            "output": raw if accepted else source,
            "score": gain,
            "candidate_count": len(candidates),
            "raw_checks": checks,
        }


def train_ranker(data, docs, run):
    features, labels, weights = [], [], []
    for row, doc in zip(data["train"], docs["train"]):
        candidates = candidate_pool(row["source"], doc)
        identity_score = sari(row["source"], row["source"], row["reference"])
        for c in candidates:
            features.append(candidate_features(row["source"], c, doc))
            labels.append(
                sari(row["source"], c["output"], row["reference"]) - identity_score
            )
            weights.append(1 / len(candidates))
    model = make_pipeline(
        DictVectorizer(sparse=False),
        HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=5,
            learning_rate=0.05,
            early_stopping=False,
            random_state=4701,
        ),
    )
    model.fit(features, labels, histgradientboostingregressor__sample_weight=weights)
    ranker = RewriteRanker(model)
    predictions = [
        ranker.predict(r["source"], d) for r, d in zip(data["dev"], docs["dev"])
    ]
    choices = []
    # Recompute with each threshold so alternative passing candidates are handled identically.
    for threshold in [0.0, 1.0, 2.0, 5.0, 10.0]:
        ranker.threshold = threshold
        if threshold:
            predictions = [
                ranker.predict(r["source"], d) for r, d in zip(data["dev"], docs["dev"])
            ]
        mean = float(
            np.mean(
                [
                    sari(r["source"], p["output"], r["reference"])
                    for r, p in zip(data["dev"], predictions)
                ]
            )
        )
        choices.append({"threshold": threshold, "dev_sari": mean})
    chosen = max(choices, key=lambda r: (r["dev_sari"], r["threshold"]))
    ranker.threshold = chosen["threshold"]
    with (Path(run) / "ranker.pkl").open("wb") as stream:
        pickle.dump(ranker, stream)
    result = {
        "feature_version": RANKER_FEATURE_VERSION,
        "training_candidates": len(labels),
        "training_pairs": len(data["train"]),
        "label": "Single-reference SARI(candidate) minus SARI(identity), not human quality",
        "selection": choices,
        "selected": chosen,
    }
    write_json(Path(run) / "ranker_training.json", result)
    return ranker, result


def edit_alignment(doc, reference):
    """Exact monotonic edit alignment, allowing an input-subject copy insertion.

    Unsupported insertions are filtered later using the train-only vocabulary.
    EOS is an explicit position so final inserted text is never silently lost.
    """
    source = [t.text for t in doc]
    target = [t.text for t in nlp()(reference)]
    tags = [["KEEP", []] for _ in range(len(source) + 1)]
    for op, a, b, c, d in SequenceMatcher(
        None,
        [t.casefold() for t in source],
        [t.casefold() for t in target],
        autojunk=False,
    ).get_opcodes():
        if op in {"delete", "replace"}:
            for i in range(a, b):
                tags[i][0] = "DELETE"
        if op in {"insert", "replace"}:
            tags[a][1].extend(target[c:d])
    subject = subject_tokens(doc)
    lower_subject = [s.casefold() for s in subject]
    for tag in tags:
        prefix = tag[1]
        if subject:
            for i in range(len(prefix) - len(subject) + 1):
                if [
                    p.casefold() for p in prefix[i : i + len(subject)]
                ] == lower_subject:
                    prefix[i : i + len(subject)] = ["<SUBJECT>"]
                    break
    return tags


def render_tags(doc, tags):
    if len(tags) != len(doc) + 1:
        raise ValueError("Edit tag count must include EOS")
    subject, tokens = subject_tokens(doc), []
    changed = False
    for i, encoded in enumerate(tags):
        action, prefix = json.loads(encoded) if isinstance(encoded, str) else encoded
        if action not in {"KEEP", "DELETE"}:
            raise ValueError(f"Unknown edit action: {action}")
        changed |= bool(prefix) or action != "KEEP"
        for word in prefix:
            tokens.extend(subject if word == "<SUBJECT>" else [word])
        if i < len(doc) and action == "KEEP":
            tokens.append(doc[i].text)
    if not changed:
        return doc.text
    out = detokenize(tokens)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m[1] + m[2].upper(), out)


def token_features(doc):
    rows = []
    for i in range(len(doc) + 1):
        f = {
            "bias": 1.0,
            "position": i / max(1, len(doc)),
            "length": len(doc),
            "BOS": i == 0,
            "EOS": i == len(doc),
        }
        for offset in [-2, -1, 0, 1, 2]:
            j = i + offset
            if 0 <= j < len(doc):
                t = doc[j]
                f.update(
                    {
                        f"{offset}:word": t.lower_,
                        f"{offset}:pos": t.pos_,
                        f"{offset}:dep": t.dep_,
                        f"{offset}:tag": t.tag_,
                        f"{offset}:punct": t.is_punct,
                        f"{offset}:stop": t.is_stop,
                    }
                )
        rows.append(f)
    return rows


class EditTagger:
    def __init__(self, model, threshold=0.0):
        self.model, self.threshold = model, threshold

    def predict(self, source, doc=None):
        doc = doc if doc is not None else nlp()(source)
        features = token_features(doc)
        tags = self.model.predict_single(features)
        raw = render_tags(doc, tags)
        marginals = self.model.predict_marginals_single(features)
        changed = [
            m[t] for t, m in zip(tags, marginals) if json.loads(t) != ["KEEP", []]
        ]
        confidence = min(changed, default=0.0)
        checks = structure_checks(source, raw, doc)
        return {
            "raw": raw,
            "output": raw
            if confidence >= self.threshold and checks["pass"]
            else source,
            "score": float(confidence),
            "tags": tags,
            "raw_checks": checks,
        }


def train_editor(data, docs, run):
    from sklearn_crfsuite import CRF

    alignments = [
        edit_alignment(d, r["reference"]) for r, d in zip(data["train"], docs["train"])
    ]
    counts = Counter(tuple(tag[1]) for tags in alignments for tag in tags if tag[1])
    # Restricted phrase vocabulary learned only from training references.
    # New lexical content is disallowed, except text copied from the input subject.
    stop = nlp().Defaults.stop_words
    allowed = {
        phrase
        for phrase, count in counts.most_common(128)
        if count >= 2
        and len(phrase) <= 6
        and all(
            w == "<SUBJECT>" or not w.isalpha() or w.casefold() in stop for w in phrase
        )
    }
    x, y, ids = [], [], []
    for row, doc, tags in zip(data["train"], docs["train"], alignments):
        if any(t[1] and tuple(t[1]) not in allowed for t in tags):
            continue
        # Do not supervise deletions of content words as safe structural edits.
        if any(
            tags[t.i][0] == "DELETE"
            and t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
            and not t.is_stop
            for t in doc
        ):
            continue
        if normalize(render_tags(doc, tags)) != normalize(row["reference"]):
            continue
        x.append(token_features(doc))
        y.append([json.dumps(t) for t in tags])
        ids.append(row["id"])
    if len(x) < 20:
        raise ValueError(
            f"Only {len(x)} representable training pairs; need at least 20"
        )
    # Copy regularization uses existing training reference components, not invented simplifications.
    copy_count = 0
    for row in data["train"][: max(1, len(x) // 4)]:
        doc = nlp()(row["reference_sentences"][0])
        x.append(token_features(doc))
        y.append([json.dumps(["KEEP", []])] * (len(doc) + 1))
        copy_count += 1
    model = CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=80,
        all_possible_transitions=True,
        min_freq=2,
        verbose=False,
    )
    model.fit(x, y)
    editor = EditTagger(model)
    predictions = [
        editor.predict(r["source"], d) for r, d in zip(data["dev"], docs["dev"])
    ]
    choices = []
    for threshold in [0.0, 0.5, 0.7, 0.85, 0.95]:
        scores = [
            sari(
                r["source"],
                p["raw"]
                if p["score"] >= threshold and p["raw_checks"]["pass"]
                else r["source"],
                r["reference"],
            )
            for r, p in zip(data["dev"], predictions)
        ]
        choices.append({"threshold": threshold, "dev_sari": float(np.mean(scores))})
    chosen = max(choices, key=lambda r: (r["dev_sari"], r["threshold"]))
    editor.threshold = chosen["threshold"]
    with (Path(run) / "editor.pkl").open("wb") as stream:
        pickle.dump(editor, stream)
    result = {
        "training_pairs": len(ids),
        "available_pairs": len(data["train"]),
        "representable_fraction": len(ids) / len(data["train"]),
        "copy_regularization_examples": copy_count,
        "insertion_vocabulary": [list(p) for p in sorted(allowed)],
        "training_ids": ids,
        "selection": choices,
        "selected": chosen,
        "limitations": "Monotonic edits and bounded train-derived insertions; no arbitrary clause reordering; alignment is weak supervision",
    }
    write_json(Path(run) / "editor_training.json", result)
    return editor, result
