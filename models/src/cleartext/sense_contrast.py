"""Candidate sense contrasts using the original contextual BERT target vector.

The candidate posterior uses the original slot, not a rewritten sentence. These
are learned ranking features; WordNet connections do not certify meaning.
"""

from functools import lru_cache

import numpy as np
from nltk.corpus import wordnet as wn

from .context_similarity import features_prompted_similarity
from .context_window import windowed_word_vector, windowed_word_vectors
from .ensemble import softmax
from .features import nlp, target_token
from .lexical import synsets
from .wsd import word_pos


CONTRAST_FEATURE_NAMES = (
    "replacement_same_mass",
    "replacement_broader_mass",
    "replacement_related_mass",
    "replacement_unlinked_mass",
    "linked_similarity_gap",
    "linked_origin_similarity_gap",
    "replacement_senses_available",
)


@lru_cache(50000)
def sense_connections(name):
    sense = wn.synset(name)
    connections = {
        other.name(): "related"
        for other in sense.verb_groups() + sense.similar_tos() + sense.also_sees()
    }
    connections.update({other.name(): "broader" for other in sense.hypernyms()})
    connections[name] = "same"
    return connections


class SenseContrastScorer:
    def __init__(self, bert):
        if not hasattr(bert, "sense_vector") or not hasattr(bert, "temperature"):
            raise ValueError("sense contrast requires the existing BERT sense model")
        self.bert = bert

    def prepare(self, names):
        """Batch missing dictionary vectors with the existing sense-vector recipe."""
        missing = sorted(set(names) - self.bert.vectors.keys() - self.bert.gloss.keys())
        inputs = [
            (
                [wn.synset(name).lemma_names()[0].replace("_", " ").split()[0], ":"]
                + wn.synset(name).definition().split(),
                [0],
            )
            for name in missing
        ]
        if inputs:
            vectors = windowed_word_vectors(inputs)
            self.bert.gloss.update(
                {name: rows[0] for name, rows in zip(missing, vectors)}
            )

    def score(self, vector, distribution, pos, candidates):
        vector = np.asarray(vector)
        if (
            vector.ndim != 1
            or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
        ):
            raise ValueError("sense contrast requires a finite unit target vector")
        if any(not np.isfinite(p) or p < 0 for p in distribution.values()) or (
            distribution and not np.isclose(sum(distribution.values()), 1.0, atol=1e-6)
        ):
            raise ValueError(
                "target sense distribution must be normalized and nonnegative"
            )
        result = []
        for candidate in candidates:
            senses = synsets(
                candidate["lemma"].lower().replace(" ", "_"), word_pos(pos)
            )
            if not distribution or not senses:
                result.append(np.zeros(len(CONTRAST_FEATURE_NAMES)))
                continue
            cosine = {
                sense.name(): float(vector @ self.bert.sense_vector(sense.name()))
                for sense in senses
            }
            posterior = dict(
                zip(cosine, softmax(list(cosine.values()), self.bert.temperature))
            )
            masses = dict.fromkeys(["same", "broader", "related", "unlinked"], 0.0)
            linked, origin_gap = set(), 0.0
            for origin, probability in distribution.items():
                connections = sense_connections(origin)
                original_similarity = float(vector @ self.bert.sense_vector(origin))
                for name, mass in posterior.items():
                    relation = connections.get(name, "unlinked")
                    joint = float(probability * mass)
                    masses[relation] += joint
                    if relation != "unlinked":
                        linked.add(name)
                        origin_gap += joint * (cosine[name] - original_similarity)
            gap = max(
                (cosine[name] for name in linked), default=min(cosine.values())
            ) - max(cosine.values())
            values = [
                masses[name] for name in ("same", "broader", "related", "unlinked")
            ]
            result.append(np.asarray(values + [gap, origin_gap, 1.0]))
        return result


def features_sense_contrast(case, sense_model=None):
    if not case["candidates"]:
        return lambda case, candidate: None
    base = features_prompted_similarity(case)
    contrast = case.get("replacement_sense_contrast")
    if contrast is None:
        if sense_model is None:
            raise ValueError(
                "sense contrast needs cached features or a configured sense model"
            )
        doc = nlp()(case["text"])
        token = target_token(doc, case["target"], case["start"])
        if token is None or token.idx != case["start"]:
            raise ValueError("sense contrast requires the exact target offset")
        vector = windowed_word_vector([token.text for token in doc], token.i)
        scorer = SenseContrastScorer(sense_model.bert)
        values = scorer.score(
            vector,
            case["sense_distribution"],
            case["target_info"][1],
            case["candidates"],
        )
        contrast = dict(
            zip([candidate["word"] for candidate in case["candidates"]], values)
        )

    def row(case, candidate):
        values = np.asarray(contrast[candidate["word"]], dtype=float)
        if (
            values.shape != (len(CONTRAST_FEATURE_NAMES),)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid candidate sense contrast features")
        return np.r_[base(case, candidate), values]

    return row


features_sense_contrast.requires_sense_distribution = True


def rewritten_sense_contrast(case, sense_model, limit=5):
    """Read at most five replacements chosen by frozen fit, without candidate labels."""
    if not 1 <= limit <= 5:
        raise ValueError("rewritten sense contrast limit must be between 1 and 5")
    if not case["candidates"]:
        return {}
    start = case.get("start")
    if (
        start is None
        or case["text"][start : start + len(case["target"])].casefold()
        != case["target"].casefold()
    ):
        raise ValueError("rewritten sense contrast requires the exact target span")
    selected = sorted(case["candidates"], key=lambda candidate: -candidate["fit"])[
        :limit
    ]
    vectors = replacement_vectors(
        case["text"],
        start,
        start + len(case["target"]),
        [candidate["word"] for candidate in selected],
    )
    scorer = SenseContrastScorer(sense_model.bert)
    scores = {
        candidate["word"]: np.zeros(len(CONTRAST_FEATURE_NAMES))
        for candidate in case["candidates"]
    }
    for vector, candidate in zip(vectors, selected):
        scores[candidate["word"]] = scorer.score(
            vector, case["sense_distribution"], case["target_info"][1], [candidate]
        )[0]
    return scores


def replacement_vectors(text, start, end, words):
    from .contextual import model
    from .complete_slot import slot_inputs
    from .context_similarity import slot_vectors

    tokenizer, encoder = model()
    inputs = slot_inputs(
        tokenizer,
        text,
        start,
        end,
        words,
        limit=min(256, encoder.config.max_position_embeddings),
    )
    return slot_vectors(inputs, tokenizer, encoder)


def features_rewritten_sense_contrast(case, sense_model=None):
    if not case["candidates"]:
        return lambda case, candidate: None
    base = features_prompted_similarity(case)
    scores = case.get("rewritten_sense_contrast")
    if scores is None:
        if sense_model is None:
            raise ValueError("rewritten sense contrast needs a configured sense model")
        scores = rewritten_sense_contrast(case, sense_model)

    def row(case, candidate):
        values = np.asarray(scores[candidate["word"]], dtype=float)
        if (
            values.shape != (len(CONTRAST_FEATURE_NAMES),)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid rewritten sense contrast features")
        return np.r_[base(case, candidate), values]

    return row


features_rewritten_sense_contrast.requires_sense_distribution = True
