"""Experimental SWORDS context validator for candidates without WordNet origins."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import ensemble as E


FEATURE_NAMES = (
    "context_vector",
    "argument_fit",
    "bigram",
    "bert_slot",
    "bert_slot_relative",
    "vector_source_candidate",
    "vector_candidate_missing",
)
SIMILARITY_FEATURE_NAMES = FEATURE_NAMES + ("replacement_similarity",)


def validate_feature_names(feature_names):
    names = tuple(feature_names)
    if names not in (FEATURE_NAMES, SIMILARITY_FEATURE_NAMES):
        raise ValueError("unknown or reordered context validator features")
    return names


def context_features(slot, candidates, feature_names=FEATURE_NAMES):
    """Frozen raw signals only. No SWORDS-trained checker or invented senses."""
    names = validate_feature_names(feature_names)
    if not candidates:
        return np.empty((0, len(names)))
    members = [
        E.ContextVectorFit(),
        E.ArgumentFit(),
        E.LanguageModelFit(),
        E.BertSlotFit(),
        E.BertSlotRelative(),
    ]
    scores = [member.score(slot, candidates) for member in members]
    vectors = E.vector_rows(slot, candidates)
    scores += [
        np.asarray([row[name] for row in vectors]) for name in FEATURE_NAMES[-2:]
    ]
    if names == SIMILARITY_FEATURE_NAMES:
        from .context_similarity import similarity_for

        token = slot.token
        scores.append(
            similarity_for(
                slot.doc.text,
                token.idx,
                token.idx + len(token),
                tuple(candidate["word"] for candidate in candidates),
            )
        )
    matrix = np.column_stack(scores)
    if not np.isfinite(matrix).all():
        raise ValueError("context validator features must be finite")
    return matrix


def context_estimator():
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=4701)
    )


def select_context_threshold(labels, probabilities, precision=0.90, minimum=20):
    """Maximum calibration coverage at the declared precision and count floor.

    This is threshold selection on reused SWORDS development judgments. It is
    not a confidence guarantee or an independent semantic evaluation.
    """
    labels, probabilities = np.asarray(labels, bool), np.asarray(probabilities, float)
    if labels.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("invalid calibration arrays")
    curve = []
    for threshold in np.round(np.arange(0.50, 1.0, 0.025), 3):
        accepted = probabilities >= threshold
        count = int(accepted.sum())
        score = float(labels[accepted].mean()) if count else None
        curve.append(
            {
                "threshold": round(float(threshold), 3),
                "accepted": count,
                "precision": score,
            }
        )
    eligible = [
        row
        for row in curve
        if row["accepted"] >= minimum and row["precision"] >= precision
    ]
    return (
        max(eligible, key=lambda row: row["accepted"])["threshold"]
        if eligible
        else None
    ), curve


def eligible_novel(row, threshold):
    """The prototype's fixed admission policy. No listed-substitute labels."""
    return (
        threshold is not None
        and row.get("context_fit", -1.0) >= threshold
        and row["validation"]["dictionary_valid"]
        and row["mechanical_checks_pass"]
        and row["complexity_delta"] > 0
        and row["wordnet_evidence"]["source_support_by_relation"]["antonym"] < 0.10
    )


def choose_novel(rows, threshold):
    eligible = [row for row in rows if eligible_novel(row, threshold)]
    return (
        max(
            eligible,
            key=lambda row: (
                row["context_fit"],
                row["complexity_delta"],
                -row["bert_rank"],
            ),
        )
        if eligible
        else None
    )
