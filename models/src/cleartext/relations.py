"""WordNet relation features for an opt-in decision experiment.

The generator's hypernym bucket also contains verb groups. These features expose
the actual edges and their breadth to the learned ranker; they do not ban edits.
Pair features use resolved lemmas, so cached development rows and live scoring
follow the same path without depending on cached candidate sense lists.
"""

from functools import lru_cache

import numpy as np
from nltk.corpus import wordnet as wn

from .generation import relation_map
from .lexical import synsets
from .refine import descendants, features_specific


RELATION_NAMES = (
    "same_sense_fraction",
    "hypernym_fraction",
    "verb_group_fraction",
    "similar_fraction",
    "hypernym_loss_mean",
    "hypernym_loss_max",
    "candidate_polysemy",
    "origin_polysemy",
    "hypernym_loss_times_sense_fit",
)


@lru_cache(50000)
def relation_values(target_lemma, pos, candidate_lemma):
    key = candidate_lemma.lower().replace(" ", "_")
    origins = next(
        (
            senses
            for name, (_, senses) in relation_map(target_lemma, pos).items()
            if name.lower() == key
        ),
        None,
    )
    if origins is None:
        raise ValueError(
            f"candidate {candidate_lemma!r} lacks WordNet provenance for {target_lemma!r}"
        )
    counts = np.zeros(4)
    losses = []
    for name in origins:
        original = wn.synset(name)
        edges = (
            [original],
            original.hypernyms(),
            original.verb_groups(),
            original.similar_tos() + original.also_sees(),
        )
        for relation, linked in enumerate(edges):
            for candidate in linked:
                if key not in {lemma.lower() for lemma in candidate.lemma_names()}:
                    continue
                counts[relation] += 1
                if relation == 1:
                    losses.append(
                        float(
                            np.log1p(descendants(candidate.name()))
                            - np.log1p(descendants(name))
                        )
                    )
    if not counts.sum():
        raise ValueError(
            f"no matching WordNet edge for {target_lemma!r} -> {candidate_lemma!r}"
        )
    fractions = counts / counts.sum()
    mean_loss, max_loss = (
        (float(np.mean(losses)), max(losses)) if losses else (0.0, 0.0)
    )
    return tuple(fractions) + (
        mean_loss,
        max_loss,
        float(np.log1p(len(synsets(key, pos)))),
        float(np.log1p(len(origins))),
    )


def features_relations(case):
    """Append relation topology to v7's features. Fit-member order is the v7 order."""
    if not case["candidates"]:
        return lambda case, candidate: None
    info = case.get("target_info")
    if info is None:
        raise ValueError("relation features require resolved target_info")
    base = features_specific(case)

    def row(case, candidate):
        values = relation_values(info[0], info[1], candidate["lemma"])
        # sense_fit measures the probability of the ORIGINAL sense, even for
        # a broader candidate. The interaction exposes that distinction.
        sense_fit = float(candidate["x"][0])
        return np.r_[base(case, candidate), values, values[4] * sense_fit]

    return row
