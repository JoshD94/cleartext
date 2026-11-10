"""WordNet evidence for unaccepted proposals, without invented target senses."""

import math

from nltk.corpus import wordnet as wn

from .candidate_validation import exact_lemma_senses
from .sense_contrast import sense_connections


def wordnet_evidence(distribution, pos, lemma):
    """Record direct connections to any candidate sense.

    Support is source-posterior mass with at least one connected candidate sense.
    It is not a candidate posterior or a probability of preserving meaning.
    Different relation totals can overlap, so they need not sum to one.
    """
    if any(not math.isfinite(p) or p < 0 for p in distribution.values()) or (
        distribution and not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("source sense distribution must be normalized and nonnegative")
    senses = exact_lemma_senses(lemma, pos)
    names = {sense.name() for sense in senses}
    links = []
    totals = dict.fromkeys(("same", "broader", "related", "narrower", "antonym"), 0.0)
    for name, probability in distribution.items():
        source = wn.synset(name)
        connections = sense_connections(name)
        relations = {
            relation: sorted(
                candidate
                for candidate in names
                if connections.get(candidate) == relation
            )
            for relation in ("same", "broader", "related")
        }
        relations["narrower"] = sorted(
            names & {sense.name() for sense in source.hyponyms()}
        )
        relations["antonym"] = sorted(
            names
            & {
                other.synset().name()
                for item in source.lemmas()
                for other in item.antonyms()
            }
        )
        for relation, connected in relations.items():
            if connected:
                totals[relation] += float(probability)
                links.append(
                    {
                        "source_sense": name,
                        "source_probability": float(probability),
                        "relation": relation,
                        "candidate_senses": connected,
                    }
                )
    return {
        "candidate_senses": [
            {"name": sense.name(), "definition": sense.definition()} for sense in senses
        ],
        "source_support_by_relation": totals,
        "links": links,
        "candidate_senses_available": bool(senses),
        "source_senses_available": bool(distribution),
        "inventory": "Exact dictionary lemma; automatic morphological alternatives excluded.",
        "semantic_certified": False,
    }
