"""Opt-in dictionary validation for BERT proposals; no acceptance decision."""

from functools import lru_cache

from lemminflect import getInflection

from .bert_candidates import MaskedBertGenerator
from .generation import TAGS
from .lexical import synsets


@lru_cache(30000)
def exact_lemma_senses(lemma, pos):
    """Exclude senses returned only through WordNet's automatic morphology.

    For example, ``found`` is an exact lemma for establishing something. Its
    automatically returned ``find`` senses belong to a different lemma.
    """
    key = lemma.casefold().replace(" ", "_")
    return tuple(
        sense
        for sense in synsets(key, pos)
        if key in {item.name().casefold() for item in sense.lemmas()}
    )


def validate_proposal(candidate, pos):
    if pos not in TAGS:
        raise ValueError("unsupported target part of speech")
    reasons = []
    senses = exact_lemma_senses(candidate["lemma"], pos)
    if not senses:
        reasons.append("no exact dictionary lemma for target part of speech")
    if candidate["word"].casefold() not in dictionary_forms(candidate["lemma"], pos):
        reasons.append("surface is not a dictionary inflection of the candidate lemma")
    return {
        "dictionary_valid": not reasons,
        "rejections": reasons,
        "exact_candidate_senses": [sense.name() for sense in senses],
        "semantic_certified": False,
    }


@lru_cache(30000)
def dictionary_forms(lemma, pos):
    return frozenset(
        form.casefold()
        for tag in TAGS[pos]
        for form in getInflection(lemma.casefold(), tag=tag, inflect_oov=False) or ()
    )


def validated_proposals(candidates, pos):
    """Keep valid proposals in their original order and preserve provenance."""
    return [
        candidate
        for candidate in candidates
        if validate_proposal(candidate, pos)["dictionary_valid"]
    ]


class DictionaryValidatedBertGenerator(MaskedBertGenerator):
    """Experimental generator with exact lemma and inflection checks.

    It deliberately keeps novel target-origin senses empty. Dictionary validity
    does not make a proposal acceptable under the pipeline's meaning policy.
    """

    name = "dictionary_validated_masked_bert_proposals"

    def generate(self, token):
        target, candidates = super().generate(token)
        return target, validated_proposals(candidates, target[1]) if target else []
