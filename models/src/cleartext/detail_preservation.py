"""Context-weighted WordNet evidence for a learned detail-preservation scorer.

These are model-weighted graph features, not semantic correctness probabilities.
A broader replacement can express a likely sense while losing its distinctions.
"""

from functools import lru_cache

import numpy as np
from nltk.corpus import wordnet as wn

from .context_similarity import features_prompted_similarity
from .refine import descendants


DETAIL_FEATURE_NAMES = (
    "same_sense_mass",
    "broader_mass",
    "unlinked_mass",
    "expected_specificity_loss",
    "sense_available",
)


@lru_cache(50000)
def sense_edge(name, lemma):
    original = wn.synset(name)
    key = lemma.lower().replace(" ", "_")
    matches = lambda sense: key in {word.lower() for word in sense.lemma_names()}
    if matches(original):
        return "same_sense", 0.0
    if any(matches(sense) for sense in original.verb_groups()):
        return "verb_group", 0.0
    if any(matches(sense) for sense in original.similar_tos() + original.also_sees()):
        return "similar", 0.0
    parents = [sense for sense in original.hypernyms() if matches(sense)]
    if parents:
        # The legacy traversal can overshoot its cap by a few siblings. Clamp
        # this new feature without changing existing runs' breadth features.
        breadth = lambda name: min(descendants(name), 20000)
        losses = [
            max(0.0, np.log1p(breadth(sense.name())) - np.log1p(breadth(name)))
            for sense in parents
        ]
        return "broader", float(min(losses))
    return "unlinked", 0.0


class DetailPreservationScorer:
    def evidence(self, distribution, candidate):
        masses = {
            "same_sense": 0.0,
            "broader": 0.0,
            "unlinked": 0.0,
            "verb_group": 0.0,
            "similar": 0.0,
        }
        loss = 0.0
        for name, probability in distribution.items():
            if not np.isfinite(probability) or probability < 0:
                raise ValueError(
                    "sense distribution must contain finite nonnegative mass"
                )
            relation, breadth = sense_edge(name, candidate["lemma"])
            masses[relation] += float(probability)
            loss += float(probability) * breadth
        total = sum(masses.values())
        if distribution and not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError("sense distribution must sum to one")
        return {
            **masses,
            "expected_specificity_loss": loss,
            "sense_available": bool(distribution),
            "semantic_certified": False,
        }

    def features(self, distribution, candidate):
        values = self.evidence(distribution, candidate)
        return np.asarray(
            [
                values[name]
                for name in (
                    "same_sense",
                    "broader",
                    "unlinked",
                    "expected_specificity_loss",
                    "sense_available",
                )
            ],
            float,
        )


def features_detail_preservation(case):
    if not case["candidates"]:
        return lambda case, candidate: None
    if "sense_distribution" not in case:
        raise ValueError(
            "detail scorer requires the full contextual sense distribution"
        )
    base = features_prompted_similarity(case)
    scorer = DetailPreservationScorer()
    return lambda case, candidate: np.r_[
        base(case, candidate), scorer.features(case["sense_distribution"], candidate)
    ]


features_detail_preservation.requires_sense_distribution = True
